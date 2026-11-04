"""Stream recovery regressions against a local CPU-only server.

Build web/ first, then run from the repository root:
uv run --locked --with playwright python web/tests/frame_stream.browser.py
"""

import asyncio
import json
from pathlib import Path
import socket
import subprocess
import tempfile

from playwright.async_api import TimeoutError as PlaywrightTimeoutError, async_playwright, expect


ROOT = Path(__file__).resolve().parents[2]
SERVER = """
from pathlib import Path
from types import SimpleNamespace
import sys
import uvicorn
from mojika.server.app import create_app
from tests.test_server import FakeEngine
config = SimpleNamespace(client_dist=Path('web/dist').resolve(), control_grace_seconds=30, fps=30)
uvicorn.run(create_app(config, FakeEngine), host='127.0.0.1', port=int(sys.argv[1]), log_level='warning')
"""
PROBE = """
window.frameSockets = [];
window.drawnFrames = 0;
const NativeSocket = window.WebSocket;
window.WebSocket = class extends NativeSocket {
  constructor(...args) {
    super(...args);
    if (String(args[0]).endsWith('/api/frames')) {
      window.frameSockets.push(this);
      this.addEventListener('message', event => { if (event.data instanceof ArrayBuffer) this.lastData = event.data; });
    }
  }
};
const draw = CanvasRenderingContext2D.prototype.drawImage;
CanvasRenderingContext2D.prototype.drawImage = function(...args) {
  if (this.canvas.classList.contains('generated-frame')) window.drawnFrames++;
  return draw.apply(this, args);
};
"""


async def recover(page):
    before = await page.evaluate("""() => {
      window.retiredSocket = window.frameSockets.at(-1);
      const before = { sockets: window.frameSockets.length, frames: window.drawnFrames };
      window.retiredSocket.dispatchEvent(new MessageEvent('message', {
        data: window.retiredSocket.lastData.slice(0, -1)
      }));
      return before;
    }""")
    try:
        await page.wait_for_function(
            "before => window.frameSockets.length > before.sockets && window.drawnFrames > before.frames + 3",
            arg=before,
        )
    except PlaywrightTimeoutError:
        print(await page.evaluate("""() => ({
            sockets: window.frameSockets.map(socket => ({state: socket.readyState, bytes: socket.lastData?.byteLength})),
            frames: window.drawnFrames, hidden: document.hidden, text: document.body.innerText
        })"""), flush=True)
        raise


async def check(url):
    async with async_playwright() as p:
        probe = await p.request.new_context()
        for _ in range(100):
            try:
                response = await probe.get(url + "/api/health")
                if (await response.json())["phase"] == "ready":
                    break
            except OSError:
                pass
            except Exception as error:
                if "ECONNREFUSED" not in str(error):
                    raise
            await asyncio.sleep(0.05)
        else:
            raise TimeoutError("Test server did not become ready")
        browser = await p.chromium.launch(headless=True)
        try:
            for mobile in (False, True):
                context = await browser.new_context(
                    viewport={"width": 390 if mobile else 1280, "height": 844},
                    is_mobile=mobile,
                    has_touch=mobile,
                )
                await context.add_init_script(PROBE)
                page = await context.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                await page.goto(url)
                await page.wait_for_function("window.drawnFrames >= 3")
                assert await page.locator(".generated-frame").evaluate(
                    "canvas => Array.from(canvas.getContext('2d').getImageData(0, 0, 1, 1).data)"
                ) == [155, 155, 155, 255]
                await page.get_by_role("button", name="配信の計測", exact=True).click()
                details = page.get_by_role("region", name="配信の計測詳細")
                await expect(details).to_be_visible()
                await expect(details.locator("p")).to_have_count(0)
                box = await details.bounding_box()
                assert 0 <= box["x"] < box["x"] + box["width"] <= page.viewport_size["width"]
                assert 0 <= box["y"] < box["y"] + box["height"] <= page.viewport_size["height"]
                toolbar = await page.get_by_role("toolbar", name="更新とヒント").bounding_box()
                self_overlap = max(0, min(box["x"] + box["width"], toolbar["x"] + toolbar["width"]) - max(box["x"], toolbar["x"])) * max(0, min(box["y"] + box["height"], toolbar["y"] + toolbar["height"]) - max(box["y"], toolbar["y"]))
                assert self_overlap == 0
                await page.keyboard.press("Escape")
                await expect(details).to_have_count(0)

                await recover(page)
                await expect(page.get_by_role("alert")).to_have_count(0)

                # Messages already queued on a retired connection cannot poison recovery.
                count = await page.evaluate("window.frameSockets.length")
                await page.evaluate("""() => window.retiredSocket.dispatchEvent(
                  new MessageEvent('message', { data: new ArrayBuffer(0) })
                )""")
                await page.wait_for_timeout(500)
                assert await page.evaluate("window.frameSockets.length") == count
                await expect(page.get_by_role("alert")).to_have_count(0)

                # Valid frames and a reconnection must not erase an operation failure.
                await page.get_by_role("button", name="操作開始", exact=True).click()
                await page.get_by_role("button", name="GPT条件を編集", exact=True).click()
                await page.get_by_role("textbox", name="GPTプロンプト").fill("reject")
                await expect(page.get_by_role("alert")).to_have_text("Invalid test prompt")
                await recover(page)
                await expect(page.get_by_role("alert")).to_have_text("Invalid test prompt")
                await page.get_by_role("button", name="操作終了", exact=True).click()
                await expect(page.get_by_role("button", name="操作開始", exact=True)).to_be_enabled()
                assert not errors, errors
                print(json.dumps({"mobile": mobile, "stream_recovery_passed": True}))
                await context.close()
        finally:
            await browser.close()
            await probe.dispose()


if __name__ == "__main__":
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with tempfile.TemporaryFile(mode="w+") as log:
        server = subprocess.Popen(
            [str(ROOT / ".venv/bin/python"), "-c", SERVER, str(port)],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            asyncio.run(check(f"http://127.0.0.1:{port}"))
            log.seek(0)
            diagnostics = log.read()
            assert diagnostics.count("Client rejected image frame") == 4, diagnostics
            assert diagnostics.count("invalid-frame bytes=") == 4, diagnostics
        except Exception:
            log.seek(0)
            print(log.read(), flush=True)
            raise
        finally:
            server.terminate()
            server.wait(timeout=10)

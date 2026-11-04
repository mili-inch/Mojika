"""Browser regressions for ordered inputs; uses a local CPU-only test server.

Run from the repository root after building web/:
uv run --locked --with playwright python web/tests/input_order.browser.py
"""

import asyncio
import io
import json
from pathlib import Path
import socket
import subprocess
import tempfile

from PIL import Image
from playwright.async_api import async_playwright, expect


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


async def slider_order(page, url):
    await page.get_by_role("button", name="更新を設定", exact=True).click()
    first, release = asyncio.Event(), asyncio.Event()
    sent = []

    async def delay(route):
        operation = route.request.post_data_json["operation"]
        if operation["kind"] == "noise_mix":
            sent.append(operation["value"])
            if len(sent) == 1:
                first.set()
                await release.wait()
        await route.continue_()

    await page.route("**/api/actions", delay)
    slider = page.get_by_role("slider", name="新規ノイズの割合", exact=True)
    await slider.press("Home")
    await asyncio.wait_for(first.wait(), 5)
    await slider.press("End")
    await page.wait_for_timeout(200)
    assert sent == [0], sent
    release.set()
    await expect(slider).to_have_attribute("aria-valuetext", "1.0")
    for _ in range(100):
        response = await page.request.post(url + "/api/session")
        if (await response.json())["controls"]["noise_mix"] == 1:
            break
        await asyncio.sleep(0.02)
    else:
        raise AssertionError("The final slider value was not applied")
    assert sent == [0, 1], sent
    await page.unroute("**/api/actions", delay)
    await page.get_by_role("button", name="モード選択に戻る").click()


async def image_then_clear(page, url):
    await page.get_by_role("button", name="CLIP条件を編集", exact=True).click()
    first, release = asyncio.Event(), asyncio.Event()
    sent = []

    async def delay(route):
        operation = route.request.post_data_json["operation"]
        sent.append(operation["kind"])
        if operation["kind"] == "clip_text":
            first.set()
            await release.wait()
        await route.continue_()

    async def image(route):
        sent.append("clip_image")
        await route.continue_()

    await page.route("**/api/actions", delay)
    await page.route("**/api/clip-image?*", image)
    await page.get_by_role("textbox", name="CLIPプロンプト").fill("山")
    await asyncio.wait_for(first.wait(), 5)
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(buffer, format="PNG")
    await page.locator('input[type="file"]').set_input_files(
        {"name": "hint.png", "mimeType": "image/png", "buffer": buffer.getvalue()}
    )
    await expect(page.get_by_role("textbox", name="CLIPプロンプト")).to_be_disabled()
    await page.get_by_role("button", name="CLIP条件を解除", exact=True).click()
    await page.wait_for_timeout(200)
    assert sent == ["clip_text"], sent
    release.set()
    await expect(
        page.get_by_role("button", name="CLIP条件を解除", exact=True)
    ).to_be_enabled()
    response = await page.request.post(url + "/api/session")
    assert (await response.json())["conditions"]["clip"] is None
    assert sent == ["clip_text", "clip_image", "clear_condition"], sent
    await page.unroute("**/api/actions", delay)
    await page.unroute("**/api/clip-image?*", image)
    await page.get_by_role("button", name="モード選択に戻る").click()


async def release_cancels_waiting(page):
    await page.get_by_role("button", name="更新を設定", exact=True).click()
    first, release = asyncio.Event(), asyncio.Event()
    sent = []

    async def delay(route):
        operation = route.request.post_data_json["operation"]
        if operation["kind"] == "noise_mix":
            sent.append(operation["value"])
            first.set()
            await release.wait()
        await route.continue_()

    await page.route("**/api/actions", delay)
    slider = page.get_by_role("slider", name="新規ノイズの割合", exact=True)
    await slider.press("Home")
    await asyncio.wait_for(first.wait(), 5)
    await slider.press("End")
    await page.wait_for_timeout(200)
    await page.get_by_role("button", name="モード選択に戻る").click()
    await page.get_by_role("button", name="操作終了", exact=True).click()
    await expect(
        page.get_by_role("button", name="操作開始", exact=True)
    ).to_be_enabled()
    release.set()
    await page.wait_for_timeout(300)
    assert sent == [0], sent
    await page.unroute("**/api/actions", delay)


async def fixed_settings(page, url):
    await page.get_by_role("button", name="更新を設定", exact=True).click()
    settings = page.get_by_role("region", name="更新設定", exact=True)
    await expect(settings.get_by_role("button")).to_have_count(0)
    await expect(settings.get_by_role("slider")).to_have_count(2)
    for label, kind, minimum, maximum in (
        ("遷移時間", "transition_seconds", 0.1, 60),
        ("新規ノイズの割合", "noise_mix", 0, 1),
    ):
        slider = page.get_by_role("slider", name=label, exact=True)
        await expect(slider).to_have_attribute("step", "0.1")
        await slider.press("Home")
        await slider.press("ArrowRight")
        await expect(slider).to_have_value(str(round(minimum + 0.1, 1)))
        await slider.press("End")
        await slider.press("ArrowLeft")
        value = round(maximum - 0.1, 1)
        await expect(slider).to_have_value(str(value))
        for _ in range(100):
            state = (await (await page.request.post(url + "/api/session")).json())
            if state["controls"][kind] == value:
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError(f"Slider value did not reach the API: {kind}")
        assert set(state["controls"]) == {"transition_seconds", "noise_auto", "noise_mix"}
    await page.get_by_role("button", name="操作ヒントを表示").click()
    await expect(page.get_by_text("ノイズと条件の変化にかける時間を調整します。", exact=True)).to_be_visible()
    await page.keyboard.press("Escape")
    await page.get_by_role("button", name="モード選択に戻る").click()
    await page.get_by_role("button", name="手描きヒントを描く", exact=True).click()
    width = page.get_by_role("slider", name="線の太さ", exact=True)
    await expect(width).to_have_attribute("step", "0.1")
    await width.press("Home")
    await width.press("ArrowRight")
    await expect(width).to_have_value("0.2")
    await width.press("End")
    await width.press("ArrowLeft")
    await expect(width).to_have_value("5.9")
    await page.get_by_role("button", name="モード選択に戻る").click()


async def hint_eraser(page, url, mobile):
    await page.get_by_role("button", name="手描きヒントを描く", exact=True).click()
    pen = page.get_by_role("button", name="ペンで描く", exact=True)
    eraser = page.get_by_role("button", name="消しゴムで消す", exact=True)
    await expect(pen).to_have_attribute("aria-pressed", "true")
    width = page.get_by_role("slider", name="線の太さ", exact=True)
    await width.press("Home")
    for _ in range(19):
        await width.press("ArrowRight")
    await expect(width).to_have_value("2")
    canvas = page.locator("canvas.hint-overlay")
    box = await canvas.bounding_box()
    touch = await page.context.new_cdp_session(page) if mobile else None

    async def state():
        return await (await page.request.post(url + "/api/session")).json()

    async def stroke(points, cancel=False):
        before = await state()
        mapped = [
            {"x": box["x"] + x * box["width"] / 16,
             "y": box["y"] + y * box["height"] / 16}
            for x, y in points
        ]
        if touch:
            await touch.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [mapped[0]]})
            for point in mapped[1:]:
                await touch.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [point]})
            await touch.send("Input.dispatchTouchEvent", {"type": "touchCancel" if cancel else "touchEnd", "touchPoints": []})
        else:
            await page.mouse.move(**mapped[0])
            await page.mouse.down()
            for point in mapped[1:]:
                await page.mouse.move(**point, steps=10)
            if cancel:
                await canvas.dispatch_event("pointercancel", {"pointerId": 1})
            await page.mouse.up()
        for _ in range(100):
            after = await state()
            if cancel or after["revision"] > before["revision"]:
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError("Hint stroke was not applied")
        if cancel:
            await page.wait_for_timeout(100)
            after = await state()
            assert after["conditions"]["hint"] == before["conditions"]["hint"]
        return after["conditions"]["hint"]

    await stroke([(4, 4), (12, 4)])
    original = await stroke([(4, 12), (12, 12)])
    assert original[4 * 16 + 8] < 100 and original[12 * 16 + 8] < 100
    await eraser.click()
    await expect(eraser).to_have_attribute("aria-pressed", "true")
    await expect(pen).to_have_attribute("aria-pressed", "false")
    await expect(page.get_by_role("slider", name="消しゴムの太さ", exact=True)).to_have_value("2")
    erased = await stroke([(8, 2), (8, 6)])
    assert erased[4 * 16 + 8] == 255
    assert erased[4 * 16 + 5] == original[4 * 16 + 5]
    assert erased[10 * 16:] == original[10 * 16:]
    await page.wait_for_function("""() => {
        const pixels = document.querySelector('canvas.hint-overlay').getContext('2d').getImageData(0, 0, 16, 16).data;
        return pixels[(4*16+8)*4+3] === 0 && pixels[(12*16+8)*4+3] > 100;
    }""")
    await pen.click()
    redrawn = await stroke([(7, 4), (9, 4)])
    assert redrawn[4 * 16 + 8] < 100
    await eraser.click()
    tapped = await stroke([(8.5, 4.5)])
    assert tapped[4 * 16 + 8] > redrawn[4 * 16 + 8]
    await stroke([(8, 10), (8, 14)], cancel=True)
    await page.get_by_role("button", name="操作ヒントを表示").click()
    await expect(page.get_by_text("なぞった部分の手描きヒントを消します。", exact=True)).to_be_visible()
    notes = await page.locator(".help-note").evaluate_all("nodes => nodes.map(node => node.getBoundingClientRect().toJSON())")
    for i, note in enumerate(notes):
        assert 0 <= note["left"] < note["right"] <= page.viewport_size["width"]
        assert 0 <= note["top"] < note["bottom"] <= page.viewport_size["height"]
        for other in notes[:i]:
            assert min(note["right"], other["right"]) <= max(note["left"], other["left"]) or min(note["bottom"], other["bottom"]) <= max(note["top"], other["top"])
    await page.keyboard.press("Escape")
    await page.get_by_role("button", name="手描きヒントを解除", exact=True).click()
    for _ in range(100):
        if (await state())["conditions"]["hint"] is None:
            break
        await asyncio.sleep(0.02)
    else:
        raise AssertionError("Full hint clear did not reset the condition")
    await pen.click()
    fresh = await stroke([(8.5, 4.5)])
    assert fresh[4 * 16 + 8] < 100
    assert all(value == 255 for value in fresh[10 * 16:])
    if touch:
        await touch.detach()
    await page.get_by_role("button", name="手描きヒントを解除", exact=True).click()
    await page.get_by_role("button", name="モード選択に戻る").click()


async def check(url):
    async with async_playwright() as p:
        probe = await p.request.new_context()
        for _ in range(100):
            try:
                response = await probe.get(url + "/api/health")
                if (await response.json())["phase"] == "ready":
                    break
            except Exception:
                pass
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
                page = await context.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                await page.goto(url)
                await page.get_by_role("button", name="操作開始", exact=True).click()
                await slider_order(page, url)
                await image_then_clear(page, url)
                await fixed_settings(page, url)
                await hint_eraser(page, url, mobile)
                await release_cancels_waiting(page)
                assert not errors, errors
                print(json.dumps({"mobile": mobile, "passed": True}))
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
        finally:
            server.terminate()
            server.wait(timeout=10)

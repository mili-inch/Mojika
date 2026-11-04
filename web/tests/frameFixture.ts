export function packet(id = 1, times = [1000, 2000, 2001, 2002]) {
  const png = Uint8Array.from(Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAAAAAB5Gfe6AAABv0lEQVR4Ae3VAREAIBACQbR/A8sa5NYGz7By3trvts/fBKAB8QQQiBfAJ4gAAvEEEIgXwAoggEA8AQTiBbACCCAQTwCBeAGsAAIIxBNAIF4AK4AAAvEEEIgXwAoggEA8AQTiBbACCCAQTwCBeAGsAAIIxBNAIF4AK4AAAvEEEIgXwAoggEA8AQTiBbACCCAQTwCBeAGsAAIIxBNAIF4AK4AAAvEEEIgXwAoggEA8AQTiBbACCCAQTwCBeAGsAAIIxBNAIF4AK4AAAvEEEIgXwAoggEA8AQTiBbACCCAQTwCBeAGsAAIIxBNAIF4AK4AAAvEEEIgXwAoggEA8AQTiBbACCCAQTwCBeAGsAAIIxBNAIF4AK4AAAvEEEIgXwAoggEA8AQTiBbACCCAQTwCBeAGsAAIIxBNAIF4AK4AAAvEEEIgXwAoggEA8AQTiBbACCCAQTwCBeAGsAAIIxBNAIF4AK4AAAvEEEIgXwAoggEA8AQTiBbACCCAQTwCBeAGsAAIIxBNAIF4AK4AAAvEEEIgXwAoggEA8AQTiBbACCCAQTwCBeAGsAAIIxBNAIF4AK4AAAvEEEIgXwAogUCfwAbOYAptpxUsYAAAAAElFTkSuQmCC', 'base64'));
  const buffer = new ArrayBuffer(52 + png.length), view = new DataView(buffer);
  view.setUint32(0, 0x504b4a4d, true);
  view.setUint32(4, id, true); view.setUint32(8, 7, true);
  times.forEach((time, index) => view.setFloat64(12 + index * 8, time, true));
  view.setUint16(44, 256, true); view.setUint16(46, 256, true);
  view.setUint32(48, png.length, true);
  new Uint8Array(buffer, 52).set(png);
  return buffer;
}

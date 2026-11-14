// Official @tensorflow/tfjs-tflite 0.0.1-alpha.10 runner, using its typed
// buffers directly: no TFJS tensor/backend dependency or network CDN.
var exports = {};
let loading;
let queue = Promise.resolve();

async function load(modelUrl, runtimeBase) {
  importScripts(new URL('tflite_web_api_client.js', runtimeBase).href);
  exports.tfweb.tflite_web_api.setWasmPath(runtimeBase);
  const runner = await exports.tfweb.TFLiteWebModelRunner.create(
    modelUrl, {numThreads: 1});
  const inputs = runner.getInputs();
  const outputs = runner.getOutputs();
  if (inputs.length !== 1 || outputs.length !== 1 ||
      inputs[0].dataType !== 'float32' || inputs[0].shape !== '15600' ||
      outputs[0].dataType !== 'float32' || outputs[0].shape !== '1024') {
    runner.cleanUp();
    throw new Error('YAMNet requires float32[15600] → float32[1024]');
  }
  return runner;
}

self.onmessage = ({data: [id, modelUrl, samples, runtimeBase]}) => {
  queue = queue.then(async () => {
    try {
      if (samples.length !== 15600 ||
          !samples.every(value => Number.isFinite(value) && Math.abs(value) <= 1)) {
        throw new Error('Invalid YAMNet waveform');
      }
      const runner = await (loading ??= load(modelUrl, runtimeBase));
      runner.getInputs()[0].data().set(samples);
      if (!runner.infer()) throw new Error('YAMNet inference failed');
      const embedding = Array.from(runner.getOutputs()[0].data());
      if (embedding.length !== 1024 || !embedding.every(Number.isFinite)) {
        throw new Error('Invalid YAMNet embedding');
      }
      self.postMessage([id, embedding, null]);
    } catch (error) {
      self.postMessage([id, null, String(error)]);
    }
  });
};

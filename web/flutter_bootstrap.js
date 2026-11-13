{{flutter_js}}
{{flutter_build_config}}

_flutter.loader.load({
  onEntrypointLoaded: async function (engineInitializer) {
    const runner = await engineInitializer.initializeEngine();
    await runner.runApp();
    document.getElementById('loading')?.remove();
  }
}).catch((error) => {
  const status = document.querySelector('#loading p');
  if (status) status.textContent = 'Unable to start: ' + error.message;
  console.error(error);
});

(function () {
  const state = {
    id: null,
    name: "",
    smiles: "",
    molblock: "",
  };

  const preview = document.getElementById("preview");
  const nameInput = document.getElementById("name");
  const smilesInput = document.getElementById("smiles");
  const molblockInput = document.getElementById("molblock");

  function render() {
    preview.textContent = JSON.stringify(state, null, 2);
  }

  function buildPayload() {
    state.name = nameInput.value.trim();
    state.smiles = smilesInput.value.trim();
    state.molblock = molblockInput.value;
    return { ...state };
  }

  function syncToBridge() {
    const payload = buildPayload();
    render();
    if (window.editorBridge && window.editorBridge.onStructureEdited) {
      window.editorBridge.onStructureEdited(JSON.stringify(payload));
    }
  }

  function loadStructure(payload) {
    state.id = payload.id ?? null;
    nameInput.value = payload.name ?? "";
    smilesInput.value = payload.smiles ?? "";
    molblockInput.value = payload.molblock ?? "";
    buildPayload();
    render();
  }

  [nameInput, smilesInput, molblockInput].forEach((element) => {
    element.addEventListener("input", syncToBridge);
  });

  new QWebChannel(qt.webChannelTransport, function (channel) {
    window.editorBridge = channel.objects.editorBridge;
    window.chemstudioEditor = {
      loadStructure,
      getPayload: buildPayload,
    };
    syncToBridge();
  });
})();


const inputVariance = document.getElementById('inputVariance');
const inputN = document.getElementById('inputN');
const epochs = document.getElementById('epochsInput');
const batchSize = document.getElementById('batchSizeInput');
var currentModel = null;
var data = null;
var tensorData = null;

const meinModellButton = document.getElementById('meinModellButton');
const overFittingButton = document.getElementById('overFittingButton');
const underFittingButton = document.getElementById('underFittingButton');
const bestFittingButton = document.getElementById('bestFittingButton');

function main(){

    data = generateData();
    runFFNN(data);
}

function initializeModel() {
    const numHiddenLayers = parseInt(document.getElementById('hiddenLayers').value);
    const neuronsPerLayer = parseInt(document.getElementById('neurons').value);
    const model = createModel(numHiddenLayers, neuronsPerLayer);
    currentModel = model;
    saveModelToServer(model);
    meinModellButton.disabled = false;
    main();
}

generateData();

async function saveModelToServer(model) {
    const saveResult = await model.save('downloads://my-model');
    // Dies wird das Modell als .json-Datei und zugehörige .bin-Dateien herunterladen.
    // Sie können diese Dateien dann auf Ihren Webserver hochladen.
}


function testCurrentModel(e){
    console.log(e.target.value);
    getCurrentModel(e.target.value).then(currentModel => {
        console.log(currentModel);
        testModel(currentModel);
    });
}

async function getCurrentModel(switchCase){
    let model;
    switch (switchCase) {   
        case 'mein-modell':
            model = currentModel;
            break;
        case 'over-fitting':
            model = null;
            break;
        case 'under-fitting':
            console.log("loading underfitting model");
            model = await tf.loadLayersModel("models/underfitting/my-model.json")
            break;
        case 'best-fitting':
            model = null;
            break;
    }
    return model;
}


//document.addEventListener('DOMContentLoaded', main);
inputVariance.addEventListener('change', generateData);
inputN.addEventListener('change', generateData)
epochs.addEventListener('change', showEpochs);
batchSize.addEventListener('change', showBatchSize);
meinModellButton.addEventListener('click', testCurrentModel);
overFittingButton.addEventListener('click', testCurrentModel);
underFittingButton.addEventListener('click', testCurrentModel);
bestFittingButton.addEventListener('click', testCurrentModel);



const inputVariance = document.getElementById('inputVariance');
const inputN = document.getElementById('inputN');
const epochs = document.getElementById('epochsInput');
const batchSize = document.getElementById('batchSizeInput');
var currentModel = null;
var data = null;
var tensorData = null;


const modelButtons = document.getElementById('model-buttons').children;
const meinModellButton = document.getElementById('my-model');


function initializeModel() {
    const numHiddenLayers = parseInt(document.getElementById('hiddenLayers').value);
    const neuronsPerLayer = parseInt(document.getElementById('neurons').value);
    const model = createModel(numHiddenLayers, neuronsPerLayer);
    currentModel = model;
    meinModellButton.disabled = false;
    data = generateData();
    runFFNN(data);
}


async function saveModelToServer(model) {
    // save model
    const saveResult = await model.save('downloads://my-model');

    // save generated data in json file
    const dataJson = JSON.stringify(data);
    saveToFile(dataJson, "data.json");

    // save parameters in json file
    const parameters = {
        hiddenLayers: parseInt(document.getElementById('hiddenLayers').value),
        neuronsPerLayer: parseInt(document.getElementById('neurons').value),
        activationFunction: document.getElementById('aktivierungsfunktionInput').value,
        inputVariance: parseFloat(inputVariance.value),
        inputN: parseInt(inputN.value),
        epochs: parseInt(epochs.value),
        batchSize: parseInt(batchSize.value),
        optimizer : document.getElementById('optimizerInput').value,
        lerningRate: parseFloat(document.getElementById('learningRateInput').value),
    };
    const parametersJson = JSON.stringify(parameters);
    saveToFile(parametersJson, "parameters.json");
}

function saveToFile(jsonData, filename) {
    const blob = new Blob([jsonData], {type: "application/json"});
    // Für IE und Edge
    if (navigator.msSaveBlob) {
        navigator.msSaveBlob(blob, filename);
    } else {
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }
}


function testCurrentModel(e){
    for (let i = 0; i < modelButtons.length; i++) {
        modelButtons[i].classList.remove('active');
    }
    e.target.classList.add('active');
    getCurrentModel(e.target.value).then(result => {
        testModel(result.model, result.data);
        showParameters(result.parameters);
    });
}

function showParameters(parameters){
    document.getElementById('hiddenLayers').value = parameters.hiddenLayers;
    document.getElementById('neurons').value = parameters.neuronsPerLayer;
    document.getElementById('aktivierungsfunktionInput').value = parameters.activationFunction;
    inputVariance.value = parameters.inputVariance;
    inputN.value = parameters.inputN;
    epochs.value = parameters.epochs;
    batchSize.value = parameters.batchSize;
    document.getElementById('optimizerInput').value = parameters.optimizer;
    document.getElementById('learningRateInput').value = parameters.lerningRate;
}
    

async function getCurrentModel(switchCase){
    let model;
    let data;

    switch (switchCase) {   
        case 'my-model':
            model = currentModel;
            break;
        case 'over-fitting':
            model = await tf.loadLayersModel("models/overfitting/my-model.json");
            data = await fetch("models/underfitting/data.json").then(response => response.json());
            parameters = await fetch("models/underfitting/parameters.json").then(response => response.json());
            break;
        case 'under-fitting':
            console.log("loading underfitting model");
            model = await tf.loadLayersModel("models/underfitting/my-model.json");
            data = await fetch("models/underfitting/data.json").then(response => response.json());
            parameters = await fetch("models/underfitting/parameters.json").then(response => response.json());
            break;
        case 'best-fitting':
            model = await tf.loadLayersModel("models/bestfitting/my-model.json");
            data = await fetch("models/underfitting/data.json").then(response => response.json());
            parameters = await fetch("models/underfitting/parameters.json").then(response => response.json());
            break;
    }

    return { model: model, data: data, parameters: parameters };
}

generateData();

//document.addEventListener('DOMContentLoaded', main);
inputVariance.addEventListener('change', generateData);
inputN.addEventListener('change', generateData)
epochs.addEventListener('change', showEpochs);
batchSize.addEventListener('change', showBatchSize);
for (let i = 0; i < modelButtons.length; i++) {
    modelButtons[i].addEventListener('click', testCurrentModel);
}



const inputVariance = document.getElementById('inputVariance');
const inputN = document.getElementById('inputN');
const epochs = document.getElementById('epochsInput');
const batchSize = document.getElementById('batchSizeInput');
var currentModel = null;
var trainedModel = null;
var data = null;
var tensorData = null;


const modelButtons = document.getElementById('model-buttons').children;
const meinModellButton = document.getElementById('my-model');
const info = document.getElementsByClassName('info');


generateData();

//document.addEventListener('DOMContentLoaded', main);
inputVariance.addEventListener('change', generateData);
inputN.addEventListener('change', generateData)
epochs.addEventListener('input', showEpochs);
batchSize.addEventListener('input', showBatchSize);
for (let i = 0; i < modelButtons.length; i++) {
    modelButtons[i].addEventListener('click', testCurrentModel);
}



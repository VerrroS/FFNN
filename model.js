
async function runFFNN(data, trainedModel){
    const values = generateData();
    visualizeData(values);
    //tfvis.show.modelSummary({name: 'Model Summary'}, trainedModel);
    tensorData = convertToTensor(data);
    const {inputs, labels} = tensorData;
    await trainedModel.compile();
    await trainedModel.trainModel(inputs, labels);
    //saveModelToServer(model);
    // Make some predictions using the model and compare them to the
    // original data
    testModel(trainedModel, data);
    for (let i = 0; i < modelButtons.length; i++) {
        modelButtons[i].classList.remove('selected');
    }
    meinModellButton.classList.add('selected');
    document.getElementById('info-box').style.display = 'none';
}


function visualizeData(values) {
  tfvis.render.scatterplot(
    { name: 'xValues v yValues' },
    { values },
    {
      xLabel: 'xValues',
      yLabel: 'yValues',
      height: 300
    }
  );
}

function generateData() {
  data = generateData();
  const values = data.map(d => ({
    x: d.x,
    y: d.y,
  }));
  return values;
}

function createModel(numHiddenLayers, neuronsPerLayer) {
  const activationFunction = document.getElementById('aktivierungsfunktionInput').value;
  const model = tf.sequential();
  model.add(tf.layers.dense({inputShape: [1], units: neuronsPerLayer, useBias: true}));
  for (let i = 0; i < numHiddenLayers; i++) {
      model.add(tf.layers.dense({units: neuronsPerLayer, activation: activationFunction, useBias: true}));
  }
  model.add(tf.layers.dense({units: 1, useBias: true}));
  return model;
}



// Funktion zum Abrufen des ausgewählten Optimizers
function getOptimizer() {
  const selectedOptimizer = document.getElementById('optimizerInput').value;
  const learningRate = document.getElementById('learningRateInput').value;

  switch (selectedOptimizer) {
      case 'sgd':
          return tf.train.sgd(learningRate);
      case 'momentum':
          return tf.train.momentum(learningRate);
      case 'adagrad':
          return tf.train.adagrad(learningRate);
      case 'adadelta':
          return tf.train.adadelta(learningRate);
      case 'adam':
          return tf.train.adam(learningRate);
      case 'adamax':
          return tf.train.adamax(learningRate);
      case 'rmsprop':
          return tf.train.rmsprop(learningRate);
      default:
          return tf.train.adam(learningRate); // Standardwert
  }
}


async function trainModel(model, inputs, labels) {
  model.compile();

  const batchSize = parseInt(document.getElementById('batchSizeInput').value);
  const epochs = parseInt(document.getElementById('epochsInput').value);

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}

function testModel(model, data) {
  const tensorData = convertToTensor(data);
  const {inputMax, inputMin, labelMin, labelMax} = tensorData;
  const predictedPoints = generatePredictions(model, inputMax, inputMin, labelMax, labelMin);
  const originalPoints = data.map(d => ({
    x: d.x, y: d.y,
  }));
  const { unverrauschteXs, unverrauschteYs } = generateCleanData();
  scatterPlot(originalPoints, predictedPoints, unverrauschteXs, unverrauschteYs);
}

function scatterPlot(originalPoints, predictedPoints, unverrauschteXs, unverrauschteYs) {
  const originalTrace = {
    x: originalPoints.map(p => p.x),
    y: originalPoints.map(p => p.y),
    mode: 'markers',
    type: 'scatter',
    name: 'Original'
  };

  const predictedTrace = {
    x: predictedPoints.map(p => p.x),
    y: predictedPoints.map(p => p.y),
    mode: 'markers',
    type: 'scatter',
    name: 'Predicted'
  };

  const unverrauschteTrace = {
    x: unverrauschteXs,
    y: unverrauschteYs,
    mode: 'lines',
    type: 'scatter',
    name: 'Unverrauscht'
  };

  Plotly.newPlot('testPlot', [originalTrace, predictedTrace, unverrauschteTrace], {
    title: 'Model Predictions vs Original Data vs Unverrauschte Funktion',
    xaxis: { title: 'xValues' },
    yaxis: { title: 'yValues' },
    height: 500
  });
}

function generateCleanData() {
  const unverrauschteXs = tf.linspace(-1, 1, 100).dataSync();
  const unverrauschteYs = unverrauschteXs.map(x => (x + 0.8) * (x - 0.2) * (x - 0.3) * (x - 0.6));
  return { unverrauschteXs, unverrauschteYs };
}

function generatePredictions(model, inputMax, inputMin, labelMax, labelMin) {
  const [xs, preds] = tf.tidy(() => {
    const xsNorm = tf.linspace(0, 1, 100);
    const predictions = model.predict(xsNorm.reshape([100, 1]));

    const unNormXs = xsNorm
      .mul(inputMax.sub(inputMin))
      .add(inputMin);

    const unNormPreds = predictions
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] };
  });
  return predictedPoints;
}

  async function getCurrentModel(switchCase){
    let model;
    let data;

    switch (switchCase) {   
        case 'my-model':
            model = trainedModel;
            document.getElementById('info-box').style.display = 'none';
            break;
        case 'over-fitting':
            model = await tf.loadLayersModel("models/overfitting/my-model.json");
            data = await fetch("models/overfitting/data.json").then(response => response.json());
            parameters = await fetch("models/overfitting/parameters.json").then(response => response.json());
            showModelInfo('over-fitting');
            break;
        case 'under-fitting':
            model = await tf.loadLayersModel("models/underfitting/my-model.json");
            data = await fetch("models/underfitting/data.json").then(response => response.json());
            parameters = await fetch("models/underfitting/parameters.json").then(response => response.json());
            showModelInfo('under-fitting');
            break;
        case 'best-fitting':
            model = await tf.loadLayersModel("models/bestfitting/my-model.json");
            data = await fetch("models/bestfitting/data.json").then(response => response.json());
            parameters = await fetch("models/bestfitting/parameters.json").then(response => response.json());
            showModelInfo('best-fitting');
            break;
    }

    return { model: model, data: data, parameters: parameters };
}


function testCurrentModel(e){
  for (let i = 0; i < modelButtons.length; i++) {
      modelButtons[i].classList.remove('selected');
  }
  e.target.classList.add('selected');
  getCurrentModel(e.target.value).then(result => {
      testModel(result.model, result.data);
      showParameters(result.parameters);
  });
}

function initializeModel() {
  const trainedModel = new Model();
  currentModel = trainedModel;
  meinModellButton.disabled = false;
  data = generateData();
  runFFNN(data, currentModel);
}

async function saveModelToServer(model) {
  const saveResult = await model.save('downloads://my-model');
  const dataJson = JSON.stringify(data);
  saveToFile(dataJson, "data.json");
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

class Model{
  constructor(){
    this.hiddenLayers = parseInt(document.getElementById('hiddenLayers').value);
    this.neuronsPerLayer = parseInt(document.getElementById('neurons').value);
    this.activationFunction = document.getElementById('aktivierungsfunktionInput').value;
    this.optimizer = document.getElementById('optimizerInput').value;
    this.learningRate = parseFloat(document.getElementById('learningRateInput').value);
    this.batchSize = parseInt(document.getElementById('batchSizeInput').value);
    this.epochs = parseInt(document.getElementById('epochsInput').value);
    this.model = this.createModel();
  }

  createModel(){
    const model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [1], units: this.neuronsPerLayer, useBias: true}));
    for (let i = 0; i < this.numHiddenLayers; i++) {
        model.add(tf.layers.dense({units: this.neuronsPerLayer, activation: this.activationFunction, useBias: true}));
    }
    model.add(tf.layers.dense({units: 1, useBias: true}));
    return model;
  }

  async compile(){
    this.model.compile({
      optimizer: this.optimizer,
      loss: tf.losses.meanSquaredError,
      metrics: ['mse', 'accuracy'],
    });
  }

  async trainModel(inputs, labels){
    return this.model.fit(inputs, labels, {
      batchSize: this.batchSize,
      epochs: this.epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
        { name: 'Training Performance' },
        ['loss', 'mse'],
        { height: 200, callbacks: ['onEpochEnd'] }
      )
    });
  }

  predict(input){
    return this.model.predict(input);
  }
}
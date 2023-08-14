
async function runFFNN(){
  data = generateData();
    const values = data.map(d => ({
        x: d.x,
        y: d.y,
    }));

    tfvis.render.scatterplot(
        {name: 'xValues v yValues'},
        {values},
        {
          xLabel: 'xValues',
          yLabel: 'yValues',
          height: 300
        }
      );
    const hiddenLayers = parseInt(document.getElementById('hiddenLayers').value);
    const neurons = parseInt(document.getElementById('neurons').value);
    const model = createModel(hiddenLayers, neurons);
    tfvis.show.modelSummary({name: 'Model Summary'}, model);

    // Convert the data to a form we can use for training.
    tensorData = convertToTensor(data);
    const {inputs, labels} = tensorData;

    // Train the model
    await trainModel(model, inputs, labels);
    saveModelToServer(model);
    // Make some predictions using the model and compare them to the
    // original data
    testModel(model, data);
    for (let i = 0; i < modelButtons.length; i++) {
        modelButtons[i].classList.remove('selected');
    }
    meinModellButton.classList.add('selected');
    document.getElementById('info-box').style.display = 'none';
}


function createModel(numHiddenLayers, neuronsPerLayer) {
  const activationFunction = document.getElementById('aktivierungsfunktionInput').value;
  const model = tf.sequential();
  // Eingabeschicht
  model.add(tf.layers.dense({inputShape: [1], units: neuronsPerLayer, useBias: true}));

  // Versteckte Schichten
  for (let i = 0; i < numHiddenLayers; i++) {
      model.add(tf.layers.dense({units: neuronsPerLayer, activation: activationFunction, useBias: true}));
  }

  // Ausgabeschicht
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
  const optimizer = getOptimizer();

  model.compile({
    optimizer: optimizer,
    loss: tf.losses.meanSquaredError,
    metrics: ['mse', 'accuracy'],
  });

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

  // Generate predictions for a uniform range of numbers between 0 and 1;
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
    return {x: val, y: preds[i]}
  });

  const originalPoints = data.map(d => ({
    x: d.x, y: d.y,
  }));

  // Daten für die unverrauschte Funktion generieren
  const unverrauschteXs = tf.linspace(-1, 1, 100).dataSync();
  const unverrauschteYs = unverrauschteXs.map(x => (x+0.8)*(x-0.2)*(x-0.3)*(x-0.6));

  // Daten für das Scatterplot vorbereiten
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

  // Scatterplot mit Plotly erstellen
  Plotly.newPlot('testPlot', [originalTrace, predictedTrace, unverrauschteTrace], {
    title: 'Model Predictions vs Original Data vs Unverrauschte Funktion',
    xaxis: { title: 'xValues' },
    yaxis: { title: 'yValues' },
    height: 500
  });
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
  const numHiddenLayers = parseInt(document.getElementById('hiddenLayers').value);
  const neuronsPerLayer = parseInt(document.getElementById('neurons').value);
  const model = createModel(numHiddenLayers, neuronsPerLayer);
  trainedModel = model;
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
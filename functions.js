function generatePlot(data)
{
    const noisyTrace = {
    x: data.map(d => d.x),
    y: data.map(d => d.y),
    mode: 'markers',
    type: 'scatter',
    name: 'Unverrauschte Daten'
  };

  const noisyData = [noisyTrace];
  const noisyLayout = {
    title: 'verrauschte Daten',
    xaxis: { title: 'x' },
    yaxis: { title: 'y' }
  };
  Plotly.newPlot('noisy-plot', noisyData, noisyLayout);
}

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
    saveModel(model);

    // Make some predictions using the model and compare them to the
    // original data
    testModel(model);
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

async function saveModel(model) {
  await model.save('localstorage://my-model');
  console.log('Model saved');
}



function convertToTensor(data) {
// Wrapping these calculations in a tidy will dispose any
// intermediate tensors.

    return tf.tidy(() => {
        // Step 1. Shuffle the data
        tf.util.shuffle(data);

        // Step 2. Convert data to Tensor
        const inputs = data.map(d => d.x)
        const labels = data.map(d => d.y);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        
        return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        // Return the min/max bounds so we can use them later.
        inputMax,
        inputMin,
        labelMax,
        labelMin,
        }
});
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
    metrics: ['mse'],
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

  function testModel(model) {
    const {inputMax, inputMin, labelMin, labelMax} = tensorData;
  
    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling
    // that we did earlier.
    const [xs, preds] = tf.tidy(() => {
  
      const xsNorm = tf.linspace(0, 1, 100);
      const predictions = model.predict(xsNorm.reshape([100, 1]));
  
      const unNormXs = xsNorm
        .mul(inputMax.sub(inputMin))
        .add(inputMin);
  
      const unNormPreds = predictions
        .mul(labelMax.sub(labelMin))
        .add(labelMin);
  
      // Un-normalize the data
      return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });
  
  
    const predictedPoints = Array.from(xs).map((val, i) => {
      return {x: val, y: preds[i]}
    });
  
    const originalPoints = data.map(d => ({
      x: d.x, y: d.y,
    }));
  
  
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

    // Scatterplot mit Plotly erstellen
    Plotly.newPlot('testPlot', [originalTrace, predictedTrace], {
      title: 'Model Predictions vs Original Data',
      xaxis: { title: 'xValues' },
      yaxis: { title: 'yValues' },
      height: 500
    });

  }

function generateData(){
    const N = document.getElementById('inputN').value; // Anzahl der zufälligen x-Werte
    const variance = document.getElementById('inputVariance').value/500;
    const data = [];
    for (let i = 0; i < N; i++) {
        x = Math.random() * 2 - 1;
        y = calculateYValue(x);
        noise = getRandomNoise(variance);
        y_noisy = y + noise;
        data.push({x: x, y: y_noisy});
    }
    generatePlot(data);
    return data;
}

function calculateYValue(x) {
  const y = (x + 0.8) * (x - 0.2) * (x - 0.3) * (x - 0.6);
  return y;
}


function getRandomNoise(variance) {
  let u = 0, v = 0;
  while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
  while(v === 0) v = Math.random();
  const number = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
  return Math.sqrt(variance)*number;
}
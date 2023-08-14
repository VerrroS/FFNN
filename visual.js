function showVariance(){
    const variance = document.getElementById('inputVariance').value;
    const varianceLabel = document.getElementById('varianceLabel');
    varianceLabel.innerHTML = variance/10;
  }
  
  function showN(){
    const N = document.getElementById('inputN').value;
    const Nlabel = document.getElementById('Nlabel');
    Nlabel.innerHTML = N;
  }
  

  function showEpochs(){
    const epochs = document.getElementById('epochsInput').value;
    const epochsLabel = document.getElementById('epochsLabel');
    epochsLabel.innerHTML = epochs;
  }

  function showBatchSize(){
    const batchSize = document.getElementById('batchSizeInput').value;
    const batchSizeLabel = document.getElementById('batchSizeLabel');
    batchSizeLabel.innerHTML = batchSize;
  }


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
    title: 'Trainingsdaten',
    xaxis: { title: 'x' },
    yaxis: { title: 'y' }
  };
  Plotly.newPlot('noisy-plot', noisyData, noisyLayout);
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
  showEpochs();
  showBatchSize();
}

function showModelInfo(model){
  document.getElementById('info-box').style.display = 'block';
  info.forEach(element => {
    if(element.id === model){
      element.style.display = 'block';
    } else {
      element.style.display = 'none';}
});
}
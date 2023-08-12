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
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

    function generateData(){
        const N = document.getElementById('inputN').value; // Anzahl der zufälligen x-Werte
        const variance = document.getElementById('inputVariance').value;
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
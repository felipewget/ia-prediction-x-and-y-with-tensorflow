async function learnLinear( sample, num_of_steps = 1 )
{

    let xs;
    let ys;
    let x           = [];
    let y           = [];
    let predictTime = sample.length;
    let model       = tf.sequential();

    sample = await sample.map( ( obj ) => {

        return {
            y: obj.price,
            x: obj.day
        }

    });

    for( let i in sample ){

        y.push( sample[i].y );
        x.push( sample[i].x );

    }

    model.add( tf.layers.dense({ units: 1, inputShape: [1]}) )

    model.compile({
        loss: "meanSquaredError",
        optimizer: "sgd"
    });

    xs = tf.tensor2d( x, [ x.length, 1 ] );
    ys = tf.tensor2d( y, [ y.length, 1 ] );

    await model.fit( xs, ys, {epochs: 600 });

    predictTime = num_of_steps + predictTime;

    let response = await model.predict( tf.tensor2d([predictTime], [1,1]) );

    return response;

}

async function convertToTensors( data, targets, percent_to_test, num_class )
{

  let num_examples = data.length;
  if( num_examples !== targets.length ){
    throw new Error("Numero de targets e diferente do numero de data");
  }

  let num_test_examples = Math.round( num_examples * percent_to_test );
  let num_train_examples = num_examples - num_test_examples;

  let x_dimension = data[0].length;

  let xs = tf.tensor2d( data, [num_examples, x_dimension ]);
  let ys = tf.oneHot( tf.tensor1d( targets ).toInt(), num_class ); // @ TODO, altera aki dps, num_class mas eo numero de classes

  let y_train = ys.slice([0, 0]  , [num_train_examples, num_class]);
  let x_train = xs.slice( [0, 0]  , [num_train_examples, x_dimension]);
  let x_test = xs.slice([num_train_examples, 0], [num_test_examples, x_dimension])
  let y_test = ys.slice([0,0],[num_test_examples, num_class]);

  return [x_train, y_train, x_test, y_test];

}

async function processModel( sample, percent_to_test, num_of_steps = 1 )
{

  let num_class = 0;
  let arr_matrizes = [];
  let arr_targets = [];
  for( let i in sample ){
    let count_indexes = sample[i].length;

    if( count_indexes > 0 ){

      let matriz_index  = ( count_indexes - 1 );
      if( !arr_matrizes[ sample[i][matriz_index] ] ){
        arr_matrizes[ sample[i][matriz_index] ] = [];
        arr_targets[sample[i][matriz_index]] = [];
        num_class++;
      }
      arr_targets[ sample[i][matriz_index] ].push( sample[i][matriz_index] )
      arr_matrizes[ sample[i][matriz_index] ].push( sample[i].slice( 0, matriz_index ) )

    }

  }

  let arr_x_train = [];
  let arr_y_train = [];
  let arr_x_test = [];
  let arr_y_test = [];

  for( let i in arr_targets ){ // Matrizes por tipo de targets

    if( arr_matrizes[i] && arr_matrizes[i].length > 0 ){

        let [ x_train, y_train, x_test, y_test ] = await convertToTensors( arr_matrizes[i], arr_targets[i], percent_to_test, num_class );

        arr_x_train.push( x_train );
        arr_y_train.push( y_train );
        arr_x_test.push( x_test );
        arr_y_test.push( y_test );

    }

  }

  let concat_axis = 0;

  return [
    tf.concat( arr_x_train, concat_axis), tf.concat( arr_y_train, concat_axis ),
    tf.concat( arr_x_test, concat_axis), tf.concat( arr_y_test, concat_axis )
  ];

}

async function trainModel( arr_x_train, arr_y_train, arr_x_test, arr_y_test, num_class, epoch ){

  let model = tf.sequential();
  let learning_rate = .01;
  let epochs = epoch;
  let optimizer = await tf.train.adam( learning_rate );
  let num

  await model.add( tf.layers.dense({ units: 10, activation: 'sigmoid', inputShape: [arr_x_train.shape[1]] }) );
  await model.add( tf.layers.dense({ units: num_class, activation: 'softmax' }));

  await model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });


  let logs_epoch = async ( epoch, logs ) => {
     console.log("Epoch: " + epoch + " | Log: " + logs.loss )
     await tf.nextFrame();
 };

  await model.fit( arr_x_train, arr_y_train, { epochs: epochs, validationData: [arr_x_test, arr_y_test], callbacks: { onEpochEnd: logs_epoch } });

  return model;

}

function hadleClickClassify( input, sample, percent_to_test, epoch, legends_classify=[] )
{

  // Classifiquei meu modelo pra começar a classifica-lo
  processModel( sample, percent_to_test ).then( function( response ){

    let [ arr_x_train, arr_y_train, arr_x_test, arr_y_test ] = response;

    let num_class = legends_classify.length;

    trainModel( arr_x_train, arr_y_train, arr_x_test, arr_y_test, num_class, epoch ).then( async function( model ){

      // model
      input = await tf.tensor2d( input, [1,input.length]);
      let predictions = await model.predict( input );
      predictions = predictions.dataSync()

      let predict = await model.predict( input ).argMax(-1).dataSync();

      console.log('-------')
      console.log( 'Classificacao: ' + legends_classify[predict] );
      console.log('-------')
      for( let i in predictions ){
        console.log( legends_classify[i] + ": " + predictions[i].toFixed(2) );
      }

    });

  })

}

function hadleClickPrediction( sample )
{

    learnLinear( sample ).then( function( response ){

        document.getElementById("response").innerText = response.get(0,0);
        console.log( response.get(0,0) );

    })

}

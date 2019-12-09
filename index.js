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

function hadleClickPrediction( sample )
{

    learnLinear( sample ).then( function( response ){

        document.getElementById("response").innerText = response.get(0,0);
        console.log( response.get(0,0) );
    
    })

}

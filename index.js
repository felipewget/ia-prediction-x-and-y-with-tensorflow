<!DOCTYPE html>
<html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.11.6"></script>

        <script>

            let sample = [
                {
                    day: 1,
                    price: 1.99,
                },
                {
                    day: 2,
                    price: 2.99,
                },
                {
                    day: 3,
                    price: 3.99,
                },
                {
                    day: 4,
                    price: 4.99,
                },
                {
                    day: 5,
                    price: 5.99,
                },
                {
                    day: 6,
                    price: 6.99,
                },
            ];

            let sample_classify = [
              [10,-15,-30,-45,-20,  0],
              [10,-25,-33,-22,5,    0],
              [5,-15,-32,-45,15,    0],
              [1,2,3,4,5,        1],
              [1,2,3,4,5,        1],
              [5,6,7,8,9,        1],
              [5,-5,13,15,15,       2],
              [-5,-14,-17,-31,-18,  2],
              [-5,-15,-16,-20,22,   2],
            ];

            let legends_classify = [
              'imagem_tipo_1',
              'imagem_tipo_2',
              'imagem_tipo_3',
            ]

            let percent_to_test = .2; // and 0.8 is to train
            let epoch = 100;

            let input = [1,2,3,4,5];

        </script>
    </head>

    <body>
        <div id="response"></div>
        <button onClick="hadleClickPrediction( sample )" >Click aki to predition</button>
        <button onClick="hadleClickClassify( input, sample_classify, percent_to_test, epoch, legends_classify )" >Click aki to classifique</button>
        <script src="index.js"></script>
    </body>
</html>

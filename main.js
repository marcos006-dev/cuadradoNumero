import * as tf from '@tensorflow/tfjs';
const inputValorCalulcar = document.getElementById('valorACalcular');
const contenedorResultado = document.getElementById('resultado');
// const mostrarVisor = document.getElementById('mostrarVisor');
let cantidadEpocas = document.getElementById('cantidadEpocas');
let modeloEntrenado;
const EPOCAS = 30;

// se establece la constante para el visor
// const surface = tfvis.visor().surface({
//   name: 'Estado del entrenamiento del modelo',
//   tab: 'Entrenamiento',
// });
// const armarGrafica = (valorX, valorY) => {
//   const trace1 = {
//     x: [valorX],
//     y: [valorY],
//     mode: 'markers',
//     // type: 'scatter',
//   };

//   // Define Data
//   const data = [trace1];

//   // Define Layout
//   const layout = {
//     xaxis: { range: [-4000, 4000], title: 'Valores de X' },
//     yaxis: { range: [-4000, 4000], title: 'Valores de Y' },
//     // title: 'House Prices vs. Size',
//   };

//   Plotly.newPlot('myPlot', data, layout);
// };

const trainUp = async () => {
  const jsxs = [];
  const jsys = [];

  // CONJUNTO DE DATOS PARA LA PRUEBA
  const dataSize = 10;
  const stepSize = 0.001;
  for (let i = 0; i < dataSize; i = i + stepSize) {
    jsxs.push(i);
    jsys.push(i * i);
  }

  // VALORES DE X
  const xs = tf.tensor(jsxs);
  // VALORES DE Y
  const ys = tf.tensor(jsys);

  // MOSTRAR PROGRESO DEL ENTRENAMIENTO
  const printCallback = {
    onEpochEnd: (epoch, log) => {
      cantidadEpocas.innerHTML = `Epocas hechas: ${epoch}/${EPOCAS} - Perdida: ${log.loss} - Precisión: ${log.acc}`;
      console.log(epoch, log);
    },
  };

  // CAPAS DEL MODELO
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: 1,
      units: 20,
      activation: 'relu',
    })
  );
  model.add(
    tf.layers.dense({
      units: 20,
      activation: 'relu',
    })
  );
  model.add(
    tf.layers.dense({
      units: 1,
    })
  );

  // PARAMETROS DEL MODELO
  model.compile({
    optimizer: 'adam',
    loss: 'meanSquaredError',
    metrics: ['accuracy'],
  });

  // ENTRENAR MODELO Y MOSTRAR SU TIEMPO
  console.time('Tiempo de Entrenamiento');
  await model.fit(xs, ys, {
    epochs: 30,
    callbacks: [
      printCallback,
      // tfvis.show.fitCallbacks(surface, ['loss', 'acc'], {
      //   name: 'Entrenamiento',
      // }),
    ],
    batchSize: 64,
  });

  inputValorCalulcar.disabled = false;
  inputValorCalulcar.focus();

  modeloEntrenado = model;
  contenedorResultado.innerHTML = 'Modelo entrenado, listo para usar';
  console.timeEnd('Training');

  // PREDECIR NUMERO
  // const next = tf.tensor([7]);
  // const answer = model.predict(next);
  // answer.print();

  // SE LIBERA DE LA MEMORIA
  // answer.dispose();
  xs.dispose();
  ys.dispose();
  // model.dispose();
};

document.addEventListener('DOMContentLoaded', () => {
  trainUp();

  // funcion que se ejecuta al presionar el boton calcular
  inputValorCalulcar.addEventListener('keyup', (event) => {
    if (event.keyCode === 13) {
      event.preventDefault();
      // convertir a numero el valor ingresado
      const valorACalcular = parseInt(inputValorCalulcar.value);
      // se realiza la prediccion
      const resultado = modeloEntrenado.predict(
        tf.tensor2d([valorACalcular], [1, 1])
      );

      // se obtiene el valor de la prediccion
      const valorResultado = resultado.dataSync();
      // console.log(valorResultado);
      // se muestra el resultado en la grafica
      // armarGrafica(valorACalcular, valorResultado[0]);
      contenedorResultado.innerHTML = `El resultado aproximado del cuadrado de ${valorACalcular} es: ${valorResultado}`;
    }
  });

  // funcion para mostrar y ocultar el visor
  // mostrarVisor.addEventListener('click', () => {
  //   tfvis.visor().toggle();
  // });
});

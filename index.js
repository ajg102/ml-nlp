require('@tensorflow/tfjs')
const tf = require('@tensorflow/tfjs-node')
const use = require('@tensorflow-models/universal-sentence-encoder')
const notes_training = require('./notes_training.json')
const notes_testing = require('./notes_testing.json')

const encodeData = data => {
    const notes = data.map(note => note.text.toLowerCase())
    const trainingData = use.load()
    .then(model => {
        return model.embed(notes)
            .then(embeddings => {
                return embeddings
            })
    })
    .catch(err => {
        console.log(err)
    })
    return trainingData
}

const outputData = tf.tensor2d(notes_training.map(note => [
    note.outcome === 'pass' ? 1 : 0,
    note.outcome === 'fail' ? 1 : 0,
    note.reason === 0 ? 1 : 0,
    note.reason === 1 ? 1 : 0,
    note.reason === 2 ? 1 : 0,
    note.reason === 3 ? 1 : 0,
    note.reason === 4 ? 1 : 0,
    note.reason === 5 ? 1 : 0,
    note.reason === 6 ? 1 : 0,
    note.reason === 7 ? 1 : 0,
    note.reason === 8 ? 1 : 0,
    note.reason === 9 ? 1 : 0,
]))

const model = tf.sequential()

model.add(tf.layers.dense({
    inputShape: [512],
    activation: 'sigmoid',
    units: 12
}))

model.add(tf.layers.dense({
    inputShape: [2],
    activation: 'sigmoid',
    units: 12
}))

model.add(tf.layers.dense({
    inputShape: [2],
    activation: 'sigmoid',
    units: 12
}))

model.compile({
    loss: 'meanSquaredError',
    optimizer: tf.train.adam(.06)
})

function run() {
    Promise.all([
        encodeData(notes_training),
        encodeData(notes_testing)
    ])
    .then(data => {
        const {
            0: trainingData,
            1: testingData 
        } = data

        model.fit(trainingData, outputData, {epochs: 200})
            .then(async history => {
                model.predict(testingData).print()
               // await model.save('file:///Users/alexg/Desktop/ML-NLP/my-model')
            })
    })
}

async function predict(note) {
    const model = await tf.loadLayersModel('file:///Users/alexg/Desktop/ML-NLP/my-model/model.json')
    const encodedNote = await encodeData([{text: note}])
    model.predict(encodedNote).print()
}

run()

// predict("Mom will encourage Max to use the child fork and give him \
// assistance in putting food on the fork and then into his mouth.\
// Next session, Max will work on improving his ability to put objects\
// into a container.")
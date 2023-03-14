import * as tf from "@tensorflow/tfjs";
import { Rank, Tensor } from "@tensorflow/tfjs";
import express from "express";
import { catchHandler } from "../../common/common.error";
import { validate_id_string } from "../../common/common.validate";
import PassModel from "../pass/pass.schema";
import PassReportModel from "../passReport/passReport.schema";

const rcsAI = express.Router();

// input: satelliteID, timeToPredict in unix time
// output: rcs at timeToPredict
rcsAI.post("/", async (req, res) => {
    try {
        const id: string = req.body.id;
        const { error } = validate_id_string.validate(id);
        if (error) throw error;
        const passList = await PassModel.find({ satellite: id });
        const reversePassList = passList.reverse();
        const passesInGrouping = [reversePassList[0]];
        for (let i = 0; i < reversePassList.length - 1; i++) {
            if (Number(new Date(reversePassList[i].end1)) - Number(new Date(reversePassList[i + 1].end1)) < 9000000) {
                passesInGrouping.push(reversePassList[i + 1]);
            } else break;
        }

        const reports = await PassReportModel.find({
            pass: passesInGrouping,
        });

        const timeRcsPair = [];
        for (let i = 0; i < passesInGrouping.length; i++) {
            if (reports.filter((report) => report.pass.toString() === passesInGrouping[i].id)[0])
                timeRcsPair.push([
                    passesInGrouping[i].end1.getTime(),
                    reports.filter((report) => report.pass.toString() === passesInGrouping[i].id)[0].rcs,
                ]);
        }

        const trainingData = timeRcsPair.map((pairs) => ({
            time: pairs[0],
            rcs: pairs[1],
        }));

        const model = createModel();
        const tensorData = convertToTensor(trainingData);
        const { inputs, labels } = tensorData;
        await trainModel(model, inputs, labels);
        // const p = predict(model, trainingData, tensorData, req.body.timeToPredict);

        const { inputMax, inputMin, labelMin, labelMax } = tensorData;
        const xsNorm = tf
            .tensor([Number(req.body.timeToPredict)])
            .sub(inputMin)
            .div(inputMax.sub(inputMin));
        const predictions = model.predict(xsNorm);
        const unNormXs = xsNorm.mul(inputMax.sub(inputMin)).add(inputMin);
        const unNormPreds = (predictions as Tensor<Rank>).mul(labelMax.sub(labelMin)).add(labelMin);
        res.send({ data: unNormPreds.dataSync()[0] });
    } catch (err) {
        catchHandler(res, err);
        return;
    }
});

function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
    model.add(tf.layers.dense({ units: 50, activation: "relu" }));
    model.add(tf.layers.dense({ units: 50, activation: "relu" }));
    model.add(tf.layers.dense({ units: 50, activation: "relu" }));
    model.add(tf.layers.dense({ units: 1, useBias: true }));
    return model;
}

function convertToTensor(
    data: {
        time: number;
        rcs: number;
    }[],
) {
    return tf.tidy(() => {
        const inputs = data.map((d) => d.time);
        const labels = data.map((d) => d.rcs);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            inputMax,
            inputMin,
            labelMax,
            labelMin,
        };
    });
}

async function trainModel(model: tf.Sequential, inputs: tf.Tensor<tf.Rank>, labels: tf.Tensor<tf.Rank>) {
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ["mse"],
    });

    const batchSize = 4;
    const epochs = 50;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
    });
}

export default rcsAI;

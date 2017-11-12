(*
F# port of the original C# example from the CNTK docs:
https://github.com/Microsoft/CNTK/blob/master/Examples/TrainingCSharp/Common/CifarResNetClassifier.cs
Currently not running, because batch normalization isn't implemented on CPU yet.
CIFAR-10 data imported directly into the /examples directory, using the Python script
supplied in the original CNTK repository.
*)

// Use the CNTK.fsx file to load the dependencies.

#load "../CNTK.fsx"
open System.Security.Cryptography.X509Certificates
open CNTK

open System
open System.IO
open System.Collections.Generic

let ConvBatchNormalizationLayer(
    input : Variable, 
    outFeatureMapCount : int, 
    kernelWidth : int, 
    kernelHeight : int, 
    hStride : int, 
    vStride : int,
    wScale : float, 
    bValue : float, 
    scValue : float, 
    bnTimeConst : int, 
    spatial : bool, 
    device : DeviceDescriptor) : Function =

        let numInputChannels = input.Shape.[input.Shape.Rank - 1]

        let convParams = 
            new Parameter(
                shape [ kernelWidth; kernelHeight; numInputChannels; outFeatureMapCount ],
                DataType.Float, 
                CNTKLib.GlorotUniformInitializer(wScale, -1, 2), 
                device)

        let convFunction = CNTKLib.Convolution(convParams, input, shape [ hStride; vStride; numInputChannels ])

        let biasParams = new Parameter(shape [NDShape.InferredDimension], float32 bValue, device, "")
        let scaleParams = new Parameter(shape [NDShape.InferredDimension], float32 scValue, device, "")
        let runningMean = new Constant(shape [NDShape.InferredDimension], float32 0.0, device)
        let runningInvStd = new Constant(shape [NDShape.InferredDimension], float32 0.0, device)
        let runningCount = Constant.Scalar(0.0f, device)
        
        CNTKLib.BatchNormalization(new Variable(convFunction), scaleParams, biasParams, runningMean, runningInvStd, runningCount,
            spatial, float bnTimeConst, 0.0, 1e-5 (* epsilon *))

let ConvBatchNormalizationReLULayer(
    input : Variable, 
    outFeatureMapCount : int, 
    kernelWidth : int, 
    kernelHeight : int, 
    hStride : int, 
    vStride : int, 
    wScale : float, 
    bValue : float, 
    scValue : float, 
    bnTimeConst : int, 
    spatial : bool, 
    device : DeviceDescriptor) : Function =

    let convBNFunction = ConvBatchNormalizationLayer(input, outFeatureMapCount, kernelWidth, kernelHeight, hStride, vStride, wScale, bValue, scValue, bnTimeConst, spatial, device)
    
    CNTKLib.ReLU(new Variable(convBNFunction))

let ProjectLayer(
    wProj : Variable, 
    input : Variable, 
    hStride : int, vStride : int, bValue : float, scValue : float, bnTimeConst : int,
    device : DeviceDescriptor) : Function =

        let outFeatureMapCount = wProj.Shape.[0]
        let b = new Parameter(shape [ outFeatureMapCount ], float32 bValue, device, "")
        let sc = new Parameter(shape [ outFeatureMapCount ], float32 scValue, device, "")
        let m = new Constant(shape [ outFeatureMapCount ], float32 0.0, device)
        let v = new Constant(shape [ outFeatureMapCount ], float32 0.0, device)
        
        let n = Constant.Scalar(float32 0.0, device)

        let numInputChannels = input.Shape.[input.Shape.Rank - 1]

        let c = 
            CNTKLib.Convolution(
                wProj, input, shape [ hStride; vStride; numInputChannels ], [ true ], [ false ])

        CNTKLib.BatchNormalization(new Variable(c), sc, b, m, v, n, true (* spatial*), float bnTimeConst, 0.0, 1e-5, false)

let GetProjectionMap(outputDim:int, inputDim:int, device:DeviceDescriptor) : Constant =

    if (inputDim > outputDim)
    then failwith "Can only project from lower to higher dimensionality"
    
    let projectionMapValues = Array.init<float32> (inputDim * outputDim) (fun _ -> float32 0)

    for i in 0 .. inputDim - 1 do
        projectionMapValues.[(i * inputDim) + i] <- (float32 1)

    let projectionMap = new NDArrayView(DataType.Float, shape [ 1; 1; inputDim; outputDim ], device)

    projectionMap.CopyFrom(
        new NDArrayView(
            shape [ 1; 1; inputDim; outputDim ], 
            projectionMapValues, 
            uint32 (projectionMapValues.Length), 
            device)
            )
        
    new Constant(projectionMap)
    
let ResNetNode(
    input : Variable, 
    outFeatureMapCount : int, 
    kernelWidth : int, kernelHeight : int, 
    wScale : float, bValue : float,
    scValue : float, bnTimeConst : int, spatial : bool, device : DeviceDescriptor) : Function =

        let c1 = ConvBatchNormalizationReLULayer(input, outFeatureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst, spatial, device)
        let c2 = ConvBatchNormalizationLayer(new Variable(c1), outFeatureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst, spatial, device)
        let p = CNTKLib.Plus(new Variable(c2), input)
        
        CNTKLib.ReLU(new Variable(p))

let ResNetNodeInc(
    input : Variable, 
    outFeatureMapCount : int, 
    kernelWidth : int, 
    kernelHeight : int, 
    wScale : float, 
    bValue : float,
    scValue : float, 
    bnTimeConst : int, 
    spatial : bool, 
    wProj : Variable, 
    device : DeviceDescriptor) : Function =

    let c1 = ConvBatchNormalizationReLULayer(input, outFeatureMapCount, kernelWidth, kernelHeight, 2, 2, wScale, bValue, scValue, bnTimeConst, spatial, device)
    let c2 = ConvBatchNormalizationLayer(new Variable(c1), outFeatureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst, spatial, device)

    let cProj = ProjectLayer(wProj, input, 2, 2, bValue, scValue, bnTimeConst, device)

    let p = CNTKLib.Plus(new Variable(c2), new Variable(cProj))
    CNTKLib.ReLU(new Variable(p))

let ResNetClassifier(
    input:Variable, 
    numOutputClasses:int, 
    device:DeviceDescriptor, 
    outputName:string) : Function =

        let convWScale = 7.07
        let convBValue = 0.

        let fc1WScale = 0.4
        let fc1BValue = 0.0

        let scValue = 1.
        let bnTimeConst = 4096

        let kernelWidth = 3
        let kernelHeight = 3

        let conv1WScale = 0.26
        let cMap1 = 16

        let conv1 = 
            ConvBatchNormalizationReLULayer(
                input, cMap1, kernelWidth, kernelHeight, 
                1, 1, conv1WScale, convBValue, scValue, 
                bnTimeConst, true (*spatial*), device)

        let rn1_1 = ResNetNode(new Variable(conv1), cMap1, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false (*spatial*), device)
        let rn1_2 = ResNetNode(new Variable(rn1_1), cMap1, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, true (*spatial*), device)
        let rn1_3 = ResNetNode(new Variable(rn1_2), cMap1, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false (*spatial*), device)

        let cMap2 = 32
        let rn2_1_wProj = GetProjectionMap(cMap2, cMap1, device)
        let rn2_1 = ResNetNodeInc(new Variable(rn1_3), cMap2, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, true (*spatial*), rn2_1_wProj, device)
        let rn2_2 = ResNetNode(new Variable(rn2_1), cMap2, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false (*spatial*), device)
        let rn2_3 = ResNetNode(new Variable(rn2_2), cMap2, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, true (*spatial*), device)

        let cMap3 = 64
        let rn3_1_wProj = GetProjectionMap(cMap3, cMap2, device)
        let rn3_1 = ResNetNodeInc(new Variable(rn2_3), cMap3, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, true (*spatial*), rn3_1_wProj, device)
        let rn3_2 = ResNetNode(new Variable(rn3_1), cMap3, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false (*spatial*), device)
        let rn3_3 = ResNetNode(new Variable(rn3_2), cMap3, kernelWidth, kernelHeight, convWScale, convBValue, scValue, bnTimeConst, false (*spatial*), device)

        // Global average pooling
        let poolW = 8
        let poolH = 8
        let poolhStride = 1
        let poolvStride = 1
        let pool = 
            CNTKLib.Pooling(
                new Variable(rn3_3), 
                PoolingType.Average,
                shape [ poolW; poolH; 1 ], 
                shape [ poolhStride; poolvStride; 1 ]
                )

        // Output DNN layer
        let outTimesParams = 
            new Parameter(
                shape [ numOutputClasses; 1; 1; cMap3 ], 
                DataType.Float,
                CNTKLib.GlorotUniformInitializer(fc1WScale, 1, 0), 
                device
                )

        let outBiasParams = 
            new Parameter(shape [ numOutputClasses ], float32 fc1BValue, device, "")

        CNTKLib.Plus(
            new Variable(CNTKLib.Times(outTimesParams, new Variable(pool))), 
            outBiasParams, 
            outputName)

let ValidateModelWithMinibatchSource(
    modelFile:string, 
    testMinibatchSource:MinibatchSource,
    imageDim:int[], 
    numClasses:int, 
    featureInputName:string, 
    labelInputName:string, 
    outputName:string,
    device:DeviceDescriptor, 
    maxCount:int
    ) =

        let model : Function = Function.Load(modelFile, device)
        let imageInput = model.Arguments.[0]
        let labelOutput = 
            model.Outputs 
            |> Seq.filter (fun o -> o.Name = outputName)
            |> Seq.exactlyOne

        let featureStreamInfo = testMinibatchSource.StreamInfo(featureInputName)
        let labelStreamInfo = testMinibatchSource.StreamInfo(labelInputName)

        let batchSize = 50

        let rec countErrors (total,errors) =

            printfn "Total: %i; Errors: %i" total errors

            let minibatchData = testMinibatchSource.GetNextMinibatch((uint32)batchSize, device)

            if (minibatchData = null || minibatchData.Count = 0)
            then (total,errors)        
            else

                let total = total + minibatchData.[featureStreamInfo].numberOfSamples

                // find the index of the largest label value
                let labelData = minibatchData.[labelStreamInfo].data.GetDenseData<float32>(labelOutput)
                let expectedLabels = 
                    labelData 
                    |> Seq.map (fun l ->                         
                        let largest = l |> Seq.max
                        l.IndexOf largest
                        )

                let inputDataMap = 
                    [
                        imageInput, minibatchData.[featureStreamInfo].data
                    ]
                    |> dataMap

                let outputDataMap = 
                    [ 
                        labelOutput, null 
                    ] 
                    |> dataMap
                    
                model.Evaluate(inputDataMap, outputDataMap, device)

                let outputData = outputDataMap.[labelOutput].GetDenseData<float32>(labelOutput)
                let actualLabels =
                    outputData 
                    |> Seq.map (fun l ->                         
                        let largest = l |> Seq.max
                        l.IndexOf largest
                        )

                let misMatches = 
                    (actualLabels,expectedLabels)
                    ||> Seq.zip
                    |> Seq.sumBy (fun (a, b) -> if a = b then 0 else 1)

                let errors = errors + misMatches

                if (int total > maxCount)
                then (total,errors)
                else countErrors (total,errors)

        countErrors (uint32 0,0)

let CifarDataFolder = __SOURCE_DIRECTORY__

let device = DeviceDescriptor.CPUDevice

let MaxEpochs = uint64 1

let imageDim = [| 32; 32; 3 |]
let numClasses = 10

let CreateMinibatchSource(
    mapFilePath : string, 
    meanFilePath : string,
    imageDims : int[], 
    numClasses : int, 
    maxSweeps : uint64) =

        let transforms = 
            List<CNTKDictionary>(
                [
                    CNTKLib.ReaderCrop("RandomSide",
                        (0, 0),
                        (float32 0.8, float32 1.0),
                        (float32 0.0, float32 0.0f),
                        (float32 1.0, float32 1.0),
                        "uniRatio")
                    CNTKLib.ReaderScale(imageDims.[0], imageDims.[1], imageDims.[2])
                    CNTKLib.ReaderMean(meanFilePath)
                ])

        let deserializerConfiguration = 
            CNTKLib.ImageDeserializer(
                mapFilePath,
                "labels", 
                uint32 numClasses,
                "features",
                transforms)

        let config = 
            new MinibatchSourceConfig(List<CNTKDictionary> [ deserializerConfiguration ])
        config.MaxSweeps <- maxSweeps

        CNTKLib.CreateCompositeMinibatchSource(config)

let minibatchSource = 
    CreateMinibatchSource(
        Path.Combine(CifarDataFolder, "train_map.txt"),
        Path.Combine(CifarDataFolder, "CIFAR-10_mean.xml"), 
        imageDim, 
        numClasses, 
        MaxEpochs)
let imageStreamInfo = minibatchSource.StreamInfo("features")
let labelStreamInfo = minibatchSource.StreamInfo("labels")

// build a model
let imageInput = CNTKLib.InputVariable(shape imageDim, imageStreamInfo.m_elementType, "Images")
let labelsVar = CNTKLib.InputVariable(shape [ numClasses ], labelStreamInfo.m_elementType, "Labels")
let classifierOutput = ResNetClassifier(imageInput, numClasses, device, "classifierOutput")

// prepare for training
let trainingLoss = CNTKLib.CrossEntropyWithSoftmax(new Variable(classifierOutput), labelsVar, "lossFunction")
// what is the 5 in the original C# code?
//let prediction = CNTKLib.ClassificationError(classifierOutput, labelsVar, 5, "predictionError")
let prediction = CNTKLib.ClassificationError(new Variable(classifierOutput), labelsVar, "predictionError")

let learningRatePerSample = new TrainingParameterScheduleDouble(0.0078125, uint32 1)
let trainer = 
    Trainer.CreateTrainer(
        classifierOutput, 
        trainingLoss, 
        prediction,
        List<Learner>( 
            [ 
                Learner.SGDLearner(classifierOutput.Parameters(), learningRatePerSample)
            ]
            )
        )

let minibatchSize = uint32 64
let outputFrequencyInMinibatches = 20
let miniBatchCount = 0

let report = progress (trainer, outputFrequencyInMinibatches)

let rec train step =

    let minibatchData = minibatchSource.GetNextMinibatch(minibatchSize, device)

    // Stop training once max epochs is reached.
    if (minibatchData.empty())
    then ignore ()
    else

        let arguments : IDictionary<Variable, MinibatchData> =
            [
                imageInput, minibatchData.[imageStreamInfo]
                labelsVar, minibatchData.[labelStreamInfo]
            ]
            |> dict

        trainer.TrainMinibatch(arguments, device) |> ignore
        
        let step = step + 1
        report step |> printer

        train step

train 0

// save the model
let imageClassifier = 
    Function.Combine(
        List<Variable>(
            [
                new Variable(trainingLoss) 
                new Variable(prediction) 
                new Variable(classifierOutput) 
            ]), 
        "ImageClassifier"
        ) 

let modelFile = "Cifar10Rest.model"
imageClassifier.Save(modelFile)

// validate the model

let testMinibatchSource = 
    CreateMinibatchSource(
        Path.Combine(CifarDataFolder, "test_map.txt"),
        Path.Combine(CifarDataFolder, "CIFAR-10_mean.xml"), 
        imageDim, numClasses, uint64 1)

ValidateModelWithMinibatchSource(
    modelFile, 
    testMinibatchSource,
    imageDim, numClasses, "features", "labels", "classifierOutput", device, 1000)

let total, errors = 
    ValidateModelWithMinibatchSource(
        modelFile, 
        testMinibatchSource,
        imageDim, 
        numClasses, 
        "features", 
        "labels", 
        "classifierOutput", 
        device,
        1000)

printfn "Total: %i / Errors: %i" total errors

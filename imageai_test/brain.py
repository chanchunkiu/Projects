from imageai.Classification import ImageClassification
import os

execution_path = os.getcwd() #current working direcotry
prediction = ImageClassification()
prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath(os.path.join(execution_path, "mobilenet_v2-b0353104.pth"))
prediction.loadModel()


predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "godzilla.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)


from mnist import MNIST
from ocr_agent import OcrAgent

print('initializing MNIST data set...')
mnist_data = MNIST('./MNIST')
mnist_data.gz = True
training_images, training_labels = mnist_data.load_training()
testing_images, testing_labels = mnist_data.load_testing()

print('learning classifier from training set...')
ocr_agent = OcrAgent.train(training_images, training_labels)

print('verifying classifier with testing set...')
overall_accuracy, class_accuracy = ocr_agent.verify(testing_images, testing_labels)
for i in range(10):
    print(f'class {i} accuracy: {class_accuracy[i] * 100}%')
print(f'overall accuracy: {overall_accuracy * 100}%')

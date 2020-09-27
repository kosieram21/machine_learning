from mnist import MNIST
from pca import *


class OcrAgent:
    def __init__(self, components, class_means):
        self.components = components
        self.class_means = class_means

    def classify(self, datum):
        projected = mdot(self.components, datum)
        classification_vector = [distance(projected, self.class_means[j]) for j in range(len(self.class_means))]
        return classification_vector.index(min(classification_vector))

    @staticmethod
    def train(data, labels, num_components):
        p = lambda subset_label: [data[i] for i in range(len(data)) if labels[i] == subset_label]
        partitions = [p(i) for i in range(max(labels) + 1)]

        class_means = [mean(partition) for partition in partitions]
        components = pca(class_means, num_components)
        class_means = [mdot(components, class_mean) for class_mean in class_means]
        return OcrAgent(components, class_means)

    def verify(self, data, labels):
        num_classes = len(self.class_means)
        correctly_classified = [0] * num_classes
        class_size = [0] * num_classes
        for i in range(len(data)):
            classification = self.classify(data[i])
            truth = labels[i]
            correctly_classified[truth] = correctly_classified[truth] if classification != truth else correctly_classified[truth] + 1
            class_size[truth] = class_size[truth] + 1

        local_accuracy = [correctly_classified[i] / class_size[i] for i in range(num_classes)]
        global_accuracy = sum(correctly_classified) / sum(class_size)
        return global_accuracy, local_accuracy


print('initializing MNIST data set...')
mnist_data = MNIST('./MNIST')
mnist_data.gz = True
training_images, training_labels = mnist_data.load_training()
testing_images, testing_labels = mnist_data.load_testing()

print('learning classifier from training set...')
ocr_agent = OcrAgent.train(training_images, training_labels, 9)

print('verifying classifier with testing set...')
overall_accuracy, class_accuracy = ocr_agent.verify(testing_images, testing_labels)
for c in range(10):
    print(f'class {c} accuracy: {class_accuracy[c] * 100}%')
print(f'overall accuracy: {overall_accuracy * 100}%')

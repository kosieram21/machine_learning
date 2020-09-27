from pca import *


class OcrAgent:
    def __init__(self, components, centroids):
        self.components = components
        self.centroids = centroids

    def classify(self, datum):
        projected = mdot(self.components, datum)
        classification_vector = [distance(projected, centroid) for centroid in self.centroids]
        return classification_vector.index(min(classification_vector))

    @staticmethod
    def train(training_data, training_labels):
        p = lambda subset_label: [training_data[i] for i in range(len(training_data)) if training_labels[i] == subset_label]
        partitions = [p(i) for i in range(max(training_labels) + 1)]

        class_means = [mean(partition) for partition in partitions]
        components = pca(class_means, len(class_means) - 1)
        centroids = [mdot(components, class_mean) for class_mean in class_means]
        return OcrAgent(components, centroids)

    def verify(self, testing_data, testing_labels):
        num_classes = len(self.centroids)
        correctly_classified = [0] * num_classes
        class_size = [0] * num_classes
        for i in range(len(testing_data)):
            classification = self.classify(testing_data[i])
            truth = testing_labels[i]
            correctly_classified[truth] = correctly_classified[truth] if classification != truth else correctly_classified[truth] + 1
            class_size[truth] = class_size[truth] + 1

        local_accuracy = [correctly_classified[i] / class_size[i] for i in range(num_classes)]
        global_accuracy = sum(correctly_classified) / sum(class_size)
        return global_accuracy, local_accuracy

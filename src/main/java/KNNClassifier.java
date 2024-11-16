import weka.core.Instances;

class KNNClassifier extends BaseClassifier {
    public KNNClassifier() {
        modelName = "KNN";
        // Initialize your KNN classifier here
    }

    @Override
    public void buildClassifier(Instances trainingData) throws Exception {
        // Implement KNN algorithm here
        throw new UnsupportedOperationException("Not implemented yet");
    }
}
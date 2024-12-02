import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

abstract class Base implements IModel {
    protected Classifier classifier;
    protected String modelName;

    @Override
    public String getModeName() {
        return modelName;
    }

//    @Override
//    public double[] predictInstance(Instance instance) throws Exception {
//        if (classifier == null) {
//            throw new IllegalStateException("Classifier not initialized. Call buildClassifier first.");
//        }
//        return classifier.distributionForInstance(instance);
//    }

    @Override
    public Classifier getClassifier() {
        if (classifier == null) {
            throw new IllegalStateException("Classifier not initialized. Call buildClassifier first.");
        }
        return classifier;
    }

    @Override
    public void buildClassifier(Instances trainingData) throws Exception {
        // Let implementing classes handle the building
    }
}
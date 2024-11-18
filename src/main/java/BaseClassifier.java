import weka.classifiers.Classifier;
import weka.core.Instance;

abstract class BaseClassifier implements IClassifier {
    protected Classifier classifier;
    protected String modelName;

    @Override
    public String getModeName() {
        return modelName;
    }

    @Override
    public double[] predictInstance(Instance instance) throws Exception {
        return classifier.distributionForInstance(instance);
    }

    @Override
    public Classifier getClassifier() {
        if (classifier == null) {
            throw new IllegalStateException("Classifier has not been initialized. Did you call buildClassifier()?");
        }
        return classifier;
    }
}

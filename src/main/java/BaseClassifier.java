import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

abstract class BaseClassifier implements IClassifier {
    protected Classifier classifier;
    protected String modelName;

    @Override
    public String getModeName(){
        return modelName;
    }

    //Predicts the class memberships for a given instance.
    @Override
    public double[] predictInstance(Instance instance) throws Exception{
        return classifier.distributionForInstance(instance);
    }

}

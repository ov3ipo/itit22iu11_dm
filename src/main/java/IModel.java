import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Classifier;

public interface IModel {
    void buildClassifier(Instances trainingData) throws Exception;
//    double[] predictInstance(Instance instance) throws Exception;
    String getModeName();
    Classifier getClassifier();
}

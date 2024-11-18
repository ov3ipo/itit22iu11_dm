import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.Classifier;

public interface IClassifier {
    void buildClassifier(Instances trainingData) throws Exception;
    double[] predictInstance(Instance instance) throws Exception;
    String getModeName();
    Classifier getClassifier(); // Add this method
}

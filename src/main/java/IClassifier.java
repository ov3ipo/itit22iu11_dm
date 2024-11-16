import weka.core.Instance;
import weka.core.Instances;

public interface IClassifier {
    void buildClassifier(Instances trainingData) throws Exception;
    double[] predictInstance(Instance instance) throws Exception;
    String getModeName();
}

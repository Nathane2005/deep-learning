package nl.vincent.rl;

import org.nd4j.linalg.dataset.DataSet;

import nl.vincent.networks.FullyConnected;
import nl.vincent.rl.datasets.XOR;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
    	FullyConnected network = new FullyConnected(2, 1, new int[] {20, 20, 20}, 0.0005, FullyConnected.OuputType.LINEAR);
    	
    	DataSet data = XOR.getDataSet();
    	
    	network.train(data, 15000);
    	
    	System.out.println("input: 0 0, output: "+Math.round(network.predict(new double[] {0, 0})[0]));
    	System.out.println("input: 1 0, output: "+Math.round(network.predict(new double[] {1, 0})[0]));
    	System.out.println("input: 0 1, output: "+Math.round(network.predict(new double[] {0, 1})[0]));
    	System.out.println("input: 1 1, output: "+Math.round(network.predict(new double[] {1, 1})[0]));
    }
}

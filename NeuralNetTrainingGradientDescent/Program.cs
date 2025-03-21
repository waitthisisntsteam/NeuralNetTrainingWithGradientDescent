using System;

namespace NeuralNetTrainingGradientDescent
{
    internal class Program
    {
        static void Main(string[] args)
        {
            ActivationErorrFormulas activationErorrFormulas = new ActivationErorrFormulas();
            ErrorFunction MeanSquared = new ErrorFunction(activationErorrFormulas.MeanSquared, activationErorrFormulas.MeanSquaredD);
            ActivationFunction TanH = new ActivationFunction(activationErorrFormulas.Sigmoid, activationErorrFormulas.SigmoidD);

            NeuralNetwork network = new NeuralNetwork([TanH], MeanSquared, 2, 2, 1);
            network.Randomize(new Random(), 0.25, 0.75);

            //XOR test
            double[][] desiredOutputs = [[1], [1], [0], [0]];
            double[][] inputs = [[1, 0], [0, 1], [0, 0], [1, 1]];

            double error = network.Train(inputs, desiredOutputs, 0.01);
            double originalError = error;
            while (true)
            {
                Console.WriteLine("Starting Error:");
                Console.WriteLine(originalError);
                Console.WriteLine("      ");

                Console.WriteLine("Current Error:      Current Outputs:");
                error = network.Train(inputs, desiredOutputs, 0.01); ;
                Console.Write(error);

                Console.Write("                   ");
                for (int i = 0; i < inputs.Length; i++)
                {
                    var output = network.Compute(inputs[i]);
                    for (int j = 0; j < output.Length; j++) Console.Write(output[j] + " ");
                }
                //Console.ReadKey();
                Console.Clear();
            }
        }
    }
}
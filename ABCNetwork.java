import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.time.LocalDateTime;
import java.util.Random;

/**
 * @author Ishan Mysore
 * @date 9/23/25
 * @file ABCNetwork.java
 * 
 * This code implements the steepest descent training algorithm for training neural networks in Java, restricted to an 
 * A-B-C architecture (A inputs, B hidden nodes, C outputs). The code separates functionality into different methods, 
 * including configuration, memory allocation, weight initialization, training, and running the network. The system supports
 * independent runtime modes for training and running, outputs full configuration and results, and handles user-selectable weight 
 * initialization (manual, randomized, or file input). Finally, this code demonstrates functionality on standard Boolean test
 * cases (AND, OR, XOR) with specified parameters.
 */
public class ABCNetwork
{
/*
* Configuration parameters (set in setConfigParams)
*/
   public static int numInput;            // number of input activations (a[k])
   public static int numHidden;           // number of hidden activations (h[j])
   public static int numOutput;           // number of output activations (F[i])
   public static boolean modeTrain;       // true = training; false = running
   public static boolean showInputTable;  // option for run report
   public static boolean showTruthTable;  // option for run report (running does not require truth table)
   public static double randLow;          // lowest value for random generation
   public static double randHigh;         // highest value for random generation
   public static double lambda;           // learning rate
   public static int maxIters;            // maximum number of iterations that the network will train on
   public static double errorThreshold;   // average error threshold (stop when average error <= this value)
   public static int numCases;            // for boolean problems: 4 cases 00,01,10,11 in this order
   public static String activation;       // the activation function (sigmoid, hyperbolic tangent, etc)
   public static boolean saveWeights;     // user-selectable option to save the weights
   public static String choosingWeights;  // LOAD, MANUAL, or RANDOM
   public static String fileNameLoad;     // file to load weights
   public static String fileNameSave;     // file to save weights
/*
* Network arrays (allocated in allocateMemory)
*/
   public static double[] a;             // a[k] input activations
   public static double[] h;             // h[j] hidden activations
   public static double[] F;             // F[i] output activations
   public static double[][] w_kj;        // w_kj: input k -> hidden j as [numInput][numHidden]
   public static double[][] w_ji;        // w_ji: hidden j -> output activations
   public static double[] theta_j;       // Θ_j (net into hidden j)
   public static double[] theta_i;       // Θ_i array
   public static double[][] delta_w_kj;  // stored Δw_kj [numInput][numHidden]
   public static double[][] delta_w_ji;  // stored Δw_ji [numHidden]     
   public static double[] psi_j;         // ψ_j = Ω_j * f'(Θ_j)
   public static double[] psi_i;         // ψ_i = w_i * f'(Θ_i)
   public static double[][] testCases;   // size [numCases][numInput]
   public static double[][] truth;       // truth table (targets) size [numCases] (training-only)
   public static double[][] outputs;     // the values that the neural network outputs after running
/*
* Output parameters (for training only)
*/
   public static double avgError;         // average training error
   public static int iterations;          // number of training iterations
   public static boolean reachedError;    // tracks whether we are under the error threshold
   public static boolean reachedMaxIter;  // tracks whether we have reached the max number of iterations
/*
* Other miscellaneous parameters
*/
   public static Random rng;              // Random Number Generator

/*
* Set hard-coded values for each of the configuration parameters listed above.
* @postcondition all the configuration parameters have been set to user values.
*/
   public static void setConfigParams()
   {
      numInput = 2;
      numHidden = 100;
      numOutput = 3;
      modeTrain = true;
      randLow = 0.1;
      randHigh = 1.5;
      lambda = 0.3;
      maxIters = 100000;
      errorThreshold = 2E-4;
      showInputTable = false;
      showTruthTable = true;
      numCases = 4;
      activation = "SIGMOID";
      choosingWeights = "RANDOM";
      saveWeights = true;
      fileNameLoad = "input_weights.bin";
      fileNameSave = "trained_weights.bin";
   } // public static void setConfigParams()

/*
* Initialize the random number generator.
* @postcondition the random number generator has been initialized.
*/
   public static void initializeRNG()
   {
      rng = new Random(); // initialize Random Number Generator
   }

/*
* Print the configuration parameters in a readable format for the user.
*/
   public static void echoConfigParams()
   {
      System.out.println("Time: " + LocalDateTime.now());   // print the current date and time
      System.out.println("=======================================================================================================");
      System.out.println("--- Configuration Parameters ---");
      System.out.println("Network Configuration: " + numInput + "-" + numHidden + "-" + numOutput);     // output the network configuration
      System.out.println("Mode: " + (modeTrain ? "TRAIN" : "RUN"));                          // output which mode we are in (training or running)
      System.out.println("Populate Method: " + choosingWeights);
      System.out.println(String.format("Random range: [%.4f, %.4f]", randLow, randHigh));

      if (modeTrain)
      {
         System.out.println(String.format("Learning Rate: %.6f", lambda));
         System.out.println("Max iterations: " + maxIters);
         System.out.println(String.format("Average error threshold: %.8f", errorThreshold));
      }

      System.out.println("Show Input Table (run): " + showInputTable);
      System.out.println("Show Truth Table (run): " + showTruthTable);
   } // public static void echoConfigParams()

/*
* Allocate memory for each of the network arrays listed above.
* @postcondition all of the arrays have memory allocated
*/
   public static void allocateMemory()
   {
      a = new double[numInput];                // inputs a[k]
      h = new double[numHidden];               // hidden activations h[j]
      F = new double[numOutput];               // output activations F[i]

      w_kj = new double[numInput][numHidden];  // w_kj[k][j]
      w_ji = new double[numHidden][numOutput]; // w_ji[j][i]
      
      outputs = new double[numCases][numOutput];
      testCases = new double[numCases][numInput];
      theta_j = new double[numHidden];
      theta_i = new double[numOutput];

      for (int j = 0; j < numCases; j++) testCases[j] = new double[numInput];

      if (modeTrain) // Training-only arrays: allocate only when training
      {  
         truth = new double[numCases][numOutput];
         psi_i = new double[numOutput];
         psi_j = new double[numHidden];
         delta_w_kj = new double[numInput][numHidden];
         delta_w_ji = new double[numHidden][numOutput];
      } // if (modeTrain)
   } // public static void allocateMemory()

/**
 * Save the current network weights and biases to a binary file.
 * The file format includes the network configuration (numInput, numHidden, numOutput)
 * followed by all weights and biases in a consistent order:
 *   - w_kj[k][j] (hidden × input)
 *   - w_ji[j][i] (hidden → output)
 *   - theta_j[j] (hidden biases)
 *   - theta_i[i] (output biases)
 *
 * @param filename the binary file to save to
 */
public static void saveWeights(String filename)
{
   try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(filename)))
   {
      dos.writeInt(numInput);
      dos.writeInt(numHidden);
      dos.writeInt(numOutput);

      for (int k = 0; k < numInput; k++) // save the hidden weights
      {
         for (int j = 0; j < numHidden; j++)
         {
            dos.writeDouble(w_kj[k][j]);
         }
      }

      for (int j = 0; j < numHidden; j++) // save the output weights
      {
         for (int i = 0; i < numOutput; i++)
         {
            dos.writeDouble(w_ji[j][i]);
         }
      }

      for (int j = 0; j < numHidden; j++) dos.writeDouble(theta_j[j]); // save the hidden biases

      for (int i = 0; i < numOutput; i++) dos.writeDouble(theta_i[i]); // save the output biases

      System.out.println("Weights saved successfully to " + filename);
   } // try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(filename)))
   catch (IOException e)
   {
      System.err.println("Error saving weights: " + e.getMessage());
   }
} // public static void saveWeights(String filename)

/**
 * Load network weights and biases from a binary file.
 * The file format must match what was written by saveWeights().
 * Configuration is validated against the current network.
 *
 * @param filename the binary file to load from
 */
public static void loadWeights(String filename)
{
   try (DataInputStream dis = new DataInputStream(new FileInputStream(filename)))
   {
      int fileNumInput = dis.readInt();
      int fileNumHidden = dis.readInt();
      int fileNumOutput = dis.readInt();

      if (fileNumInput != numInput || fileNumHidden != numHidden || fileNumOutput != numOutput)
      {
         throw new IllegalStateException(
               String.format("Weight file config %d-%d-%d does not match current config %d-%d-%d",
                  fileNumInput, fileNumHidden, fileNumOutput, numInput, numHidden, numOutput));
      }

      for (int k = 0; k < numInput; k++) // load hidden weights
      {
         for (int j = 0; j < numHidden; j++)
         {
            w_kj[k][j] = dis.readDouble();
         }
      }

      for (int j = 0; j < numHidden; j++) // load output weights
      {
         for (int i = 0; i < numOutput; i++)
         {
            w_ji[j][i] = dis.readDouble();
         }
      }

      for (int j = 0; j < numHidden; j++) theta_j[j] = dis.readDouble(); // load hidden biases
      
      for (int i = 0; i < numOutput; i++) theta_i[i] = dis.readDouble(); // load output biases

      System.out.println("Weights loaded successfully from " + filename);
   } // try (DataInputStream dis = new DataInputStream(new FileInputStream(filename)))
   catch (IOException e)
   {
      System.err.println("Error loading weights: " + e.getMessage());
   }
} // public static void loadWeights(String filename)

/**
* Populate values for each of the arrays for which memory was allocated earlier.
* Also set the weights manually/randomize them, depending on the corresponding parameters.
* If both preloadWeights and populateRandom are false, set the weights to zero.
*/
   public static void populateArrays()
   {
      setInputTable();

      if (modeTrain) setTruthTable();

/*
* Input the weights according to value of user-selectable option.
*/
      if (choosingWeights.equals("LOAD"))
      {
         loadWeights(fileNameLoad);
      }
      else if (choosingWeights.equals("MANUAL"))
      {
         setWeightsManually();
      }
      else
      {
         randomizeWeights(randLow, randHigh);
      }
   } // public static void populateArrays()

   public static void setInputTable()
   {
      testCases[0][0] = 0.0;
      testCases[0][1] = 0.0;
      testCases[1][0] = 0.0;
      testCases[1][1] = 1.0;
      testCases[2][0] = 1.0;
      testCases[2][1] = 0.0;
      testCases[3][0] = 1.0;
      testCases[3][1] = 1.0;
   } // public static void setInputTable()

   public static void setTruthTable()
   {
      truth[0][0] = 0.0;
      truth[1][0] = 0.0;
      truth[2][0] = 0.0;
      truth[3][0] = 1.0;
      truth[0][1] = 0.0;
      truth[1][1] = 1.0;
      truth[2][1] = 1.0;
      truth[3][1] = 1.0;
      truth[0][2] = 0.0;
      truth[1][2] = 1.0;
      truth[2][2] = 1.0;
      truth[3][2] = 0.0;
   } // public static void setTruthTable()

/**
* Manual weight initializer that works for any A-B-1.
* If (A,B) match a known case (like 2-2-1 OR), we load hand-crafted weights.
* Otherwise, we fall back to small deterministic values.
*/
   public static void setWeightsManually()
   {
      if (numInput == 2 && numHidden == 2)            // if we have a 2-2-C network
      {
         w_kj[0][0] = 1.0; w_kj[1][0] = 1.0;          // k=0,1 -> j=0
         w_kj[0][1] = -1.0; w_kj[1][1] = 1.0;         // k=0,1 -> j=1
         w_ji[0][0] = 1.0; w_ji[1][0] = 1.0;
         w_ji[0][1] = -1.0; w_ji[1][1] = 1.0;
      } // if (numInput == 2 && numHidden == 2)
      else if (numInput == 2 && numHidden == 3)       // if we have a 2-3-C network
      {
         w_kj[0][0] = 5.0;  w_kj[1][0] = 5.0;
         w_kj[0][1] = -5.0; w_kj[1][1] = 5.0;
         w_kj[0][2] = 5.0;  w_kj[1][2] = -5.0;
         w_ji[0][0] = 5.0; w_ji[1][0] = -5.0;
         w_ji[0][1] = -5.0; w_ji[1][1] = 5.0;
         w_ji[0][2] = 5.0; w_ji[1][2] = 5.0;
      } // else if (numInput == 2 && numHidden == 3)
      else if (numInput == 2 && numHidden == 5)       // if we have a 2-5-C network
      {
         w_kj[0][0] = 20.0; w_kj[1][0] = 20.0;
         w_kj[0][1] = -20.0; w_kj[1][1] = -20.0;
         w_kj[0][2] = 20.0; w_kj[1][2] = -20.0;
         w_kj[0][3] = -20.0; w_kj[1][3] = 20.0;
         w_kj[0][4] = 20.0; w_kj[1][4] = 20.0;
         w_ji[0][0] = -20.0; w_ji[1][0] = -20.0;
         w_ji[0][1] = 20.0; w_ji[1][1] = 20.0;
         w_ji[0][2] = -20.0; w_ji[1][2] = -20.0;
         w_ji[0][3] = 20.0; w_ji[1][3] = 20.0;
      } // else if (numInput == 2 && numHidden == 5)
      else                                            // set the weights to small deterministic values
      {
         for (int k = 0; k < numInput; k++)
         {
            for (int j = 0; j < numHidden; j++)
            {
               w_kj[k][j] = 0.1 * (j + 1) + 0.05 * (k + 1);
            }
         }
         for (int j = 0; j < numHidden; j++)
         {
            for (int i = 0; i < numOutput; i++)
            {
               w_ji[j][i] = 0.1 * (j + 1);
            }
         }
      } // else
   } // public static void setWeightsManually()

/**
* Select random weights for the network, with each weight ranging from the low to high values passed in as parameters.
* Calls the randRange method below to select a random value for each weight in the arrays.
* @param low the lowest possible value for the weights
* @param high the highest possible value for the weights
*/
   public static void randomizeWeights(double low, double high)
   {
      for (int k = 0; k < numInput; k++)
      {
         for (int j = 0; j < numHidden; j++)
         {
            w_kj[k][j] = randRange(low, high);
         }
      }
      
      for (int j = 0; j < numHidden; j++)
      {
         for (int i = 0; i < numOutput; i++)
         {
            w_ji[j][i] = randRange(low, high);
         }
      }
   } // public static void randomizeWeights(double low, double high)

/**
* Select a random number from low to high.
* @param low the lowest possible value we want to select
* @param high the highest possible value we want to select
* @return the random value generated
*/
   public static double randRange(double low, double high)
   {
      return low + rng.nextDouble() * (high - low);
   }

/**
* Calculates the activation function (sigmoid) for an input x.
* @param x the input that we want to calculate the sigmoid of.
* @return the sigmoid function for x
*/
   public static double sigmoid(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   }

/**
* Calculates the derivative of the sigmoid function for an input x.
* @param x the input that we want to calculate the sigmoid of.
* @return the derivative of the sigmoid function for x
*/
   public static double sigmoidDeriv(double x)
   {
      double sig = sigmoid(x);
      return sig * (1.0 - sig);
   }

   public static double activation(double x)
   {
      return sigmoid(x);
   }

   public static double activationDeriv(double x)
   {
      return sigmoidDeriv(x);
   }

   public static void setUpRunActivations(double[] input)
   {
      for (int k = 0; k < numInput; k++)
      {
         a[k] = input[k]; // copy inputs into a[k]
      }
   }

/**
* Run (single forward pass). This method only runs the network once.
* It does not depend on truth table and does no training allocations.
* Coded directly to the design document.
*/
   public static double[] runOnce()
   {
/*
* compute Θ_j and h[j]
*/ 
      for (int j = 0; j < numHidden; j++)
      {
         double sum = 0.0;
         for (int k = 0; k < numInput; k++)
         {
            sum += a[k] * w_kj[k][j]; // w_kj[k][j]
         }
         theta_j[j] = sum;
         h[j] = activation(theta_j[j]);
      } // for (int j = 0; j < numHidden; j++)
/*
* compute Θ_i and F
*/
      for (int i = 0; i < numOutput; i++)
      {
         double sumi = 0.0;
         for (int j = 0; j < numHidden; j++)
         {
            sumi += h[j] * w_ji[j][i];
         }
         theta_i[i] = sumi;
         F[i] = activation(theta_i[i]);
      } // for (int i = 0; i < numOutput; i++)
      return F;
   } // public static double[] runOnce()

   public static void setUpActivations(int n)
   {
      for (int k = 0; k < numInput; k++)
      {
         a[k] = testCases[n][k]; // a[k] = input; setting input activations
      }
   }

/**
* Train (gradient descent). Training is implemented exactly as the four-step algorithm
* in the design document.
*
* @precondition in training mode; truth and F are populated.
*/
   public static double trainOnce(int n, double sumError)
   {
      for (int i = 0; i < numOutput; i++)
      {
         double omega_i = truth[n][i] - F[i];
         psi_i[i] = omega_i * activationDeriv(theta_i[i]);
         double caseError = 0.5 * omega_i * omega_i;
         sumError += caseError;
      }

      for (int j = 0; j < numHidden; j++)
      {
         double omega_j = 0.0;
         for (int i = 0; i < numOutput; i++)
         {
            omega_j += psi_i[i] * w_ji[j][i];
         }
         psi_j[j] = omega_j * activationDeriv(theta_j[j]);
      }
/**
* Compute Δw - both hidden and output
*/
      for (int j = 0; j < numHidden; j++)
      {
         for (int i = 0; i < numOutput; i++)
         {
            delta_w_ji[j][i] = h[j] * lambda * psi_i[i];
         }
      }

      for (int k = 0; k < numInput; k++)
      {
         for (int j = 0; j < numHidden; j++)
         {
            delta_w_kj[k][j] = a[k] * lambda * psi_j[j];
         }
      }
/**
* Update the weights
*/
      for (int j = 0; j < numHidden; j++)
      {
         for (int i = 0; i < numOutput; i++)
         {
            w_ji[j][i] += delta_w_ji[j][i];     // update weights (output)
         }        
      }

      for (int k = 0; k < numInput; k++)
      {
         for (int j = 0; j < numHidden; j++)
         {
            w_kj[k][j] += delta_w_kj[k][j];     // update weights (hidden)
         }
      }
      return sumError;
   } // public static double trainOnce(int n, double sumError)

   public static void trainAllTestCases()
   {
      while (!reachedError && !reachedMaxIter) // keep going until we are under the error threshold or until max iterations
      {
         iterations++;
         double sumError = 0.0;
         for (int t = 0; t < numCases; t++)          // loop over test cases
         {
            setUpActivations(t);                     // set up training
            F = runOnce();                           // Run a forward pass of the network
            sumError = trainOnce(t, sumError);
         } // for (int t = 0; t < numCases; t++)
         avgError = sumError / (double) numCases;
/*
* check stopping criteria
*/
         if (avgError <= errorThreshold) reachedError = true;
         if (iterations >= maxIters) reachedMaxIter = true;
      } // while (!reachedError && !reachedMaxIter)
   } // public static void trainAllTestCases()

/*
* When in mode train, prints the output results of training (error, number of iterations, reasons for stopping).
*/
   public static void printTrainError()
   {
/*
* If running without training: we can still run preloaded or random weights
*/
      if (modeTrain)
      {
/*
* Report training results (reporting not part of train() itself)
*/
         System.out.println("--- Training Exit Information ---");
         if (reachedError) System.out.println("Reason: Error threshold reached (avg error <= " + errorThreshold + ")");
         if (reachedMaxIter) System.out.println("Reason: Maximum iterations reached (" + maxIters + ")");
         System.out.println("Iterations: " + iterations);
         System.out.println(String.format("Average error: %.8f", avgError));
         System.out.println("----------------------------------");
      } // if (modeTrain)
   } // public static void trainAndPrintError()

/*
* Run all test cases and produce reports.
*/
   public static void runAllTestCases()
   {
/*
* Loop over the number of test cases and run the network for each test case.
*/
      for (int t = 0; t < numCases; t++)
      {
         setUpRunActivations(testCases[t]);
         runOnce(); // run the network
         for (int i = 0; i < numOutput; i++)
         {
            outputs[t][i] = F[i];
         }
      } // for (int t = 0; t < numCases; t++)
   } // public static void runAllTestCases()

/**
* When in mode run, prints the output results (input table, truth table, output values upon user's request)
*/
   public static void printResults()
   {
      System.out.println("--- Run Results ---");
/**
* Print each value in the input table (if the user wants to show the input table)
*/
      if (showInputTable)
      {
         System.out.println("Input Table:"); // this prints the inputs in order
         for (int t = 0; t < numCases; t++)
         {
            for (int k = 0; k < numInput; k++) System.out.print((int) testCases[t][k] + " ");
            System.out.println();
         }
      } // if (showInputTable)
/**
* Print each value in the truth table (if the user want to show the truth table)
*/
      if (showTruthTable && truth != null)
      {
         System.out.println("Truth Table:");
         for (int t = 0; t < numCases; t++)
         {
            System.out.println(String.format("%d %d -> %d %d %d", 
            (int) testCases[t][0], (int) testCases[t][1], (int) truth[t][0], (int) truth[t][1], (int) truth[t][2]));
         }
      } // if (showTruthTable && truth != null)
/**
* Prints the output values.
*/
      System.out.println("Actual Outputs:");
      for (int t = 0; t < numCases; t++)
      {
         System.out.println(String.format("Input: %d %d -> Output: %.6f %.6f %.6f", 
         (int) testCases[t][0], (int) testCases[t][1], outputs[t][0], outputs[t][1], outputs[t][2]));
      }
      System.out.println("--------------------");
   } // public static void printResults()

/**
* Puts it all together!
*/
   public static void main(String[] args)
   {
      setConfigParams();
      echoConfigParams();
      initializeRNG();
      allocateMemory();
      populateArrays();

      if (modeTrain)
      {
         trainAllTestCases();
         printTrainError();
         if (saveWeights) saveWeights(fileNameSave);
      }
      
      runAllTestCases();
      printResults();
   } // public static void main(String[] args)
} // public class ABCNetwork
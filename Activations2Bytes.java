import java.io.*;

public class Activations2Bytes {

    public static void main(String[] args) {
        if (args.length != 2) {
            System.out.println("Usage: java Activations2Bytes input.act output.bin");
            return;
        }

        String inFile  = args[0];
        String outFile = args[1];

        long count = 0;

        try (
            DataInputStream in = new DataInputStream(
                new BufferedInputStream(new FileInputStream(inFile))
            );
            DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(outFile))
            )
        ) {
            while (true) {
                try {
                    // Read activation (0.0 – 1.0)
                    double val = in.readDouble();

                    // Clamp just in case
                    if (val < 0.0) val = 0.0;
                    if (val > 1.0) val = 1.0;

                    // Scale to 0–255
                    int gray = (int)Math.round(val * 255.0);

                    out.writeByte(gray);
                    count++;

                } catch (EOFException eof) {
                    break;
                }
            }

            System.out.printf(
                "Converted %d activations -> %d bytes (%s → %s)\n",
                count, count, inFile, outFile
            );

        } catch (IOException e) {
            System.err.println("File error: " + e.getMessage());
        }
    }
}

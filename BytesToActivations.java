import java.io.*;
import java.nio.file.*;

/**
 * BytesToActivations.java
 *
 * Converts 8-bit grayscale byte images into
 * double-precision activation files in [0.0, 1.0].
 *
 * Usage:
 *   java BytesToActivations input.bin width height output.act
 */
public class BytesToActivations {

    public static void main(String[] args) throws IOException {

        if (args.length != 4) {
            System.out.println("Usage: java BytesToActivations <input.bin> <width> <height> <output.act>");
            return;
        }

        String inputFile  = args[0];
        int width         = Integer.parseInt(args[1]);
        int height        = Integer.parseInt(args[2]);
        String outputFile = args[3];

        // Enforce minimum size constraint
        if (Math.min(width, height) < 256) {
            System.err.println("ERROR: Image must be at least 256 pels on the short side.");
            return;
        }

        byte[] bytes = Files.readAllBytes(Paths.get(inputFile));

        if (bytes.length != width * height) {
            System.err.println("ERROR: File size does not match width × height.");
            return;
        }

        FileOutputStream fos = new FileOutputStream(outputFile);
        DataOutputStream dos = new DataOutputStream(fos);

        // Convert byte → double activation
        for (int i = 0; i < bytes.length; i++) {
            int unsignedVal = bytes[i] & 0xFF;
            double activation = unsignedVal / 255.0;
            dos.writeDouble(activation);
        }

        dos.close();
        fos.close();
    }
}

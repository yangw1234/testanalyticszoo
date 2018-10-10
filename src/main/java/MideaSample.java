import com.intel.analytics.zoo.pipeline.inference.AbstractInferenceModel;
import com.intel.analytics.zoo.pipeline.inference.JTensor;

import java.awt.image.ColorConvertOp;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.io.IOException;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.util.List;
import java.util.ArrayList;


public class MideaSample {
    public static void main(String[] args) throws IOException{
        AbstractInferenceModel model = new AbstractInferenceModel(){};
        String modelPath = "/home/yang/sources/datasets/midea/tfnet";
        String imagePath = "/home/yang/sources/datasets/midea/0000041.jpg";
        model.loadTF(modelPath, 1, 1, true);
        BufferedImage image = ImageIO.read(new File(imagePath));
        if (image.getType() != BufferedImage.TYPE_INT_RGB) {
            ColorConvertOp op = new ColorConvertOp(null);
            BufferedImage rgbimage = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_INT_RGB);
            op.filter(image, rgbimage);
            image = rgbimage;
        }
        float[] floatImage = ImageToList(image);
        int [] shape = new int[]{image.getHeight(), image.getWidth(), 3};
        JTensor single_image = new JTensor(floatImage, shape, false);
        List<JTensor> inputs = new ArrayList<JTensor>();
        inputs.add(single_image);
        List<List<JTensor>> mInputs = new ArrayList<List<JTensor>>();
        mInputs.add(inputs);
        List<List<JTensor>> outputs = model.predict(mInputs);
        System.out.println(outputs);

    }

    private static float[] ImageToList(BufferedImage image){

        int[] pixels = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
        float[] result = new float[pixels.length*3];
        for (int i = 0; i < pixels.length; i++) {

            int data = pixels[i];
            result[i * 3] = ((float) (data & 0xff));
            result[i * 3 + 1] = ((float) (data >> 8 & 0xff));
            result[i * 3 +2] = ((float) (data >> 16 & 0xff));
        }
        return result;
    }
}

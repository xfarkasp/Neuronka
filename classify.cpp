#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "model.h"

int main(int arc, char** argv)
{
    std::string loc = "D:\\Downloads\\jablko_test2.jpg";

    // Load image with OpenCV.
    cv::Mat img = cv::imread(loc);
    cv::resize(img, img, cv::Size(64, 64), 0, 0);

    // Convert the image and label to a tensor.
    torch::Tensor img_tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kByte);
    img_tensor = img_tensor.permute({0, 3, 1, 2}); // convert to CxHxW
    img_tensor = img_tensor.to(torch::kF32);

    // Load the model.
    ConvNet model(3/*channel*/, 64/*height*/, 64/*width*/);
    torch::load(model, "../best_model.pt");

    // Predict the probabilities for the classes.
    torch::Tensor log_prob = model(img_tensor);
    torch::Tensor prob = torch::exp(log_prob);

    printf("Probability of being\n\
    an apple = %.2f percent\n\
    a banana = %.2f percent\n", prob[0][0].item<float>()*100., prob[0][1].item<float>()*100.);

    return 0;
}

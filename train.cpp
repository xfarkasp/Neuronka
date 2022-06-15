#include <torch/torch.h>

#include "custom_dataset.h"
#include "model.h"
using namespace cv;

// Convert cv::Mat to at::Tensor
at::Tensor matToTensor(const cv::Mat& image)
{
    // Copy cv::Mat data to tensor
    // Convert the image and label to a tensor.
    // Here we need to clone the data, as from_blob does not change the ownership of the underlying memory,
    // which, therefore, still belongs to OpenCV. If we did not clone the data at this point, the memory
    // would be deallocated after leaving the scope of this get method, which results in undefined behavior.
    auto tensor = torch::from_blob(image.data, { image.rows, image.cols, image.channels() }, /*torch::kUInt8*/torch::kByte).clone();

    // Reorder data in tensor
    tensor = tensor.permute({ 2, 0, 1 });

    // Convert data to float and normalize
    tensor = tensor.to(torch::kF32).div(UCHAR_MAX);

    return tensor;
}

int main()
{
    VideoCapture cap(1);
    int volba = 0;
    
    while (true) {
        std::cout << "Train = 1 \nClasify=2\nKoniec=3\nVolba: ";
        std::cin >> volba;
        if (volba == 1) {
            // Load the model.
            ConvNet model(3/*channel*/, 64/*height*/, 64/*width*/);

            // Generate your data set. At this point you can add transforms to you data set, e.g. stack your
            // batches into a single tensor.
            std::string file_names_csv = "D:\\Downloads\\libtorch_custom_dataset-master\\libtorch_custom_dataset-master\\file_names.csv";
            auto data_set = CustomDataset(file_names_csv).map(torch::data::transforms::Stack<>());

            // Generate a data loader.
            int64_t batch_size = 32;
            auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                data_set,
                batch_size);

            // Chose and optimizer.
            torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));

            // Train the network.
            int64_t n_epochs = 10;
            int64_t log_interval = 10;
            int dataset_size = data_set.size().value();

            // Record best loss.
            float best_mse = std::numeric_limits<float>::max();

            for (int epoch = 1; epoch <= n_epochs; epoch++) {

                // Track loss.
                size_t batch_idx = 0;
                float mse = 0.; // mean squared error
                int count = 0;

                for (auto& batch : *data_loader) {
                    auto imgs = batch.data;
                    auto labels = batch.target.squeeze();

                    imgs = imgs.to(torch::kF32);
                    labels = labels.to(torch::kInt64);

                    optimizer.zero_grad();
                    auto output = model(imgs);
                    auto loss = torch::nll_loss(output, labels);

                    loss.backward();
                    optimizer.step();

                    mse += loss.template item<float>();

                    batch_idx++;
                    if (batch_idx % log_interval == 0)
                    {
                        std::printf(
                            "\rTrain Epoch: %d/%ld [%5ld/%5d] Loss: %.4f",
                            epoch,
                            n_epochs,
                            batch_idx * batch.data.size(0),
                            dataset_size,
                            loss.template item<float>());
                    }

                    count++;
                }

                mse /= (float)count;
                printf(" Mean squared error: %f\n", mse);

                if (mse < best_mse)
                {
                    torch::save(model, "../best_model.pt");
                    best_mse = mse;
                }
            }
        }
        else if (volba == 2) {
            while(true){
                cv::Mat img, imgGray, imgCrop;
                cv::Mat imgResize;

                cap.read(img);
                //cv::flip(img,img, 1);

                cv::Rect roi(220, 140, 200, 200);
                imgCrop = img(roi);
                imshow("cropp", imgCrop);
           
                cv::resize(imgCrop, imgCrop, cv::Size(64, 64), 0, 0);
                torch::Tensor img_tensor = torch::from_blob(imgCrop.data, { 1, imgCrop.rows, imgCrop.cols, 3 }, torch::kByte);
                img_tensor = img_tensor.permute({ 0, 3, 1, 2 }); // convert to CxHxW
                img_tensor = img_tensor.to(torch::kF32);

                // Load the model.
                ConvNet model(3/*channel*/, 64/*height*/, 64/*width*/);
                torch::load(model, "../best_model.pt");

                // Predict the probabilities for the classes.
                torch::Tensor log_prob = model(img_tensor);
                torch::Tensor prob = torch::exp(log_prob);
                printf("Probability of being\n\
                    an apple = %.2f percent\n\
                    a banana = %.2f percent\n\
                    an orange = %.2f percent\n\
                    ", prob[0][0].item<float>() * 100., prob[0][1].item<float>() * 100., prob[0][2].item<float>() * 100.);
                std::string ovocie = " ";
                if (prob[0][0].item<float>() * 100. > prob[0][1].item<float>() * 100. && prob[0][0].item<float>() * 100. > prob[0][2].item<float>() * 100.)
                    ovocie = "jablko";
                else if(prob[0][1].item<float>() * 100.> prob[0][0].item<float>() * 100. && prob[0][1].item<float>() * 100. > prob[0][2].item<float>() * 100.)
                    ovocie = "banan";
                else if (prob[0][2].item<float>() * 100. > prob[0][0].item<float>() * 100. && prob[0][2].item<float>() * 100. > prob[0][1].item<float>() * 100.)
                    ovocie = "pomaranc";


                rectangle(img, Point(220, 140), Point(440, 340), Scalar(255, 255, 255), 3);
                putText(img, "na obrazku je: " + ovocie, Point(220, 110), FONT_HERSHEY_DUPLEX, 0.7, Scalar(0, 255, 0), 2);
                imshow("image", img);


                waitKey(1);
            }
            
        }
        else if (volba = 3)
            break;
    }
    
    return 0;
}
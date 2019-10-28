import com.jfoenix.controls.JFXTextArea;
import com.jfoenix.controls.JFXTextField;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.Pane;

import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ResourceBundle;

public class AppController implements Initializable {

    @Override
    public void initialize(URL location, ResourceBundle resources) {}

    @FXML
    private ImageView img_view, imgview_hr, imgview_ndcg;

    @FXML
    private Label title, label_dataset;

    @FXML
    private Pane trainingPane1, modelsPane, datasetsPane, resultsPane, aboutPane, traningPane2;

    @FXML
    private RadioButton rd_ml100k, rd_ml1m, rd_yelp, rd_twnu, rd_tnormal, rd_gmf, rd_hmlp, rd_nhf, rd_nhf_woutpretrain, rd_nhf_pretrain,
                        rd_mlp, rd_ncf, rd_ncf_pretrain, rd_ncf_woutpretrain;

    @FXML
    private JFXTextField jtxt_numneg, jtxt_savepath, jtxt_gmf_emb, jtxt_gmf_epochs, jtxt_hmlp_layers, jtxt_hmlp_epochs,
            jtxt_hmlp_emb, jtxt_gmf_batch, jtxt_hmlp_batch, jtxt_nhf_gmf_emb, jtxt_nhf_hmlp_emb, jtxt_nhf_layers, jtxt_nhf_epochs,
            jtxt_nhf_facts, jtxt_nhf_batch, jtxt_nhf_nbfacts, jtxt_hmlp_nbfacts, jtxt_patience, jtxt_patience2,
            jtxt_mlp_layers, jtxt_mlp_epochs, jtxt_mlp_emb, jtxt_mlp_batch, jtxt_mlp_nbfacts,
            jtxt_ncf_gmf_emb, jtxt_ncf_mlp_emb, jtxt_ncf_layers, jtxt_ncf_epochs, jtxt_ncf_batch, jtxt_ncf_nbfacts;

    @FXML
    private ComboBox cb_gmf_optimizer, cb_hmlp_optimizer, cb_nhf_submodels_optimizer, cb_nhf_optimizer, cb_mlp_optimizer, cb_ncf_optimizer, cb_ncf_submodels_optimizer;

    @FXML
    private CheckBox check_estop, check_estop2;

    @FXML
    private JFXTextArea area_hr, area_ndcg, area_models;

    private String path;

    //ON MOUSE CLICKED EVENTS
    //panes
    public void models() {
        resultsPane.setVisible(false);
        aboutPane.setVisible(false);
        trainingPane1.setVisible(false);
        traningPane2.setVisible(false);
        datasetsPane.setVisible(false);
        modelsPane.setVisible(true);
    }

    public void datasets() {
        resultsPane.setVisible(false);
        aboutPane.setVisible(false);
        trainingPane1.setVisible(false);
        traningPane2.setVisible(false);
        modelsPane.setVisible(false);
        datasetsPane.setVisible(true);
    }

    public void training() {
        modelsPane.setVisible(false);
        resultsPane.setVisible(false);
        aboutPane.setVisible(false);
        datasetsPane.setVisible(false);
        traningPane2.setVisible(false);
        trainingPane1.setVisible(true);
    }

    public void results() {
        modelsPane.setVisible(false);
        trainingPane1.setVisible(false);
        aboutPane.setVisible(false);
        datasetsPane.setVisible(false);
        traningPane2.setVisible(false);
        resultsPane.setVisible(true);
    }

    public void about() {
        modelsPane.setVisible(false);
        trainingPane1.setVisible(false);
        resultsPane.setVisible(false);
        traningPane2.setVisible(false);
        datasetsPane.setVisible(false);
        aboutPane.setVisible(true);
    }

    public void next_training(){
        modelsPane.setVisible(false);
        trainingPane1.setVisible(false);
        resultsPane.setVisible(false);
        datasetsPane.setVisible(false);
        aboutPane.setVisible(false);
        traningPane2.setVisible(true);
    }

    //early stopping text event
    public void setPatience(){  //Early stopping function
        if(check_estop.isSelected()) //estop enabled
            jtxt_patience.setDisable(false);
        else  //estop disabled
            jtxt_patience.setDisable(true);
    }

    public void setPatience2(){  //Early stopping function
        if(check_estop2.isSelected()) //estop enabled
            jtxt_patience2.setDisable(false);
        else  //estop disabled
            jtxt_patience2.setDisable(true);
    }

    //Empty panes and text fields
    public void empty_gmf(){
        rd_gmf.setSelected(false);
        jtxt_gmf_emb.setText("");
        jtxt_gmf_batch.setText("");
        jtxt_gmf_epochs.setText("");
    }

    public void empty_hmlp(){
        rd_hmlp.setSelected(false);
        jtxt_hmlp_layers.setText("");
        jtxt_hmlp_batch.setText("");
        jtxt_hmlp_emb.setText("");
        jtxt_hmlp_epochs.setText("");
        jtxt_hmlp_nbfacts.setText("");
    }

    public void empty_nhf(){
        rd_nhf.setSelected(false);
        rd_nhf_woutpretrain.setSelected(false);
        rd_nhf_pretrain.setSelected(false);
        jtxt_nhf_epochs.setText("");
        jtxt_nhf_batch.setText("");
        jtxt_nhf_nbfacts.setText("");
        jtxt_nhf_gmf_emb.setText("");
        jtxt_nhf_hmlp_emb.setText("");
        jtxt_nhf_layers.setText("");
    }

    public void empty_mlp(){
        rd_mlp.setSelected(false);
        jtxt_mlp_batch.setText("");
        jtxt_mlp_emb.setText("");
        jtxt_mlp_epochs.setText("");
        jtxt_mlp_layers.setText("");
        jtxt_mlp_nbfacts.setText("");
    }

    public void empty_ncf(){
        rd_ncf.setSelected(false);
        rd_ncf_pretrain.setSelected(false);
        rd_ncf_woutpretrain.setSelected(false);
        jtxt_ncf_nbfacts.setText("");
        jtxt_ncf_epochs.setText("");
        jtxt_ncf_gmf_emb.setText("");
        jtxt_ncf_mlp_emb.setText("");
        jtxt_ncf_layers.setText("");
        jtxt_ncf_batch.setText("");
    }

    //disable jtxt for each model
    public void disable_gmf(){
        jtxt_gmf_emb.setDisable(true);
        jtxt_gmf_batch.setDisable(true);
        jtxt_gmf_epochs.setDisable(true);
    }

    public void disable_hmlp(){
        jtxt_hmlp_layers.setDisable(true);
        jtxt_hmlp_batch.setDisable(true);
        jtxt_hmlp_emb.setDisable(true);
        jtxt_hmlp_epochs.setDisable(true);
        jtxt_hmlp_nbfacts.setDisable(true);
    }

    public void disable_nhf(){
        rd_nhf_woutpretrain.setDisable(true);
        rd_nhf_pretrain.setDisable(true);
        jtxt_nhf_epochs.setDisable(true);
        jtxt_nhf_batch.setDisable(true);
        jtxt_nhf_nbfacts.setDisable(true);
        jtxt_nhf_gmf_emb.setDisable(true);
        jtxt_nhf_hmlp_emb.setDisable(true);
        jtxt_nhf_layers.setDisable(true);
    }

    public void disable_mlp(){
        jtxt_mlp_batch.setDisable(true);
        jtxt_mlp_emb.setDisable(true);
        jtxt_mlp_epochs.setDisable(true);
        jtxt_mlp_layers.setDisable(true);
        jtxt_mlp_nbfacts.setDisable(true);
    }

    public void disable_ncf(){
        rd_ncf_pretrain.setDisable(true);
        rd_ncf_woutpretrain.setDisable(true);
        jtxt_ncf_nbfacts.setDisable(true);
        jtxt_ncf_epochs.setDisable(true);
        jtxt_ncf_gmf_emb.setDisable(true);
        jtxt_ncf_mlp_emb.setDisable(true);
        jtxt_ncf_layers.setDisable(true);
        jtxt_ncf_batch.setDisable(true);
    }

    //enable jtxt for each model
    public void enable_gmf(){
        jtxt_gmf_emb.setDisable(false);
        jtxt_gmf_batch.setDisable(false);
        jtxt_gmf_epochs.setDisable(false);
    }

    public void enable_hmlp(){
        jtxt_hmlp_layers.setDisable(false);
        jtxt_hmlp_batch.setDisable(false);
        jtxt_hmlp_emb.setDisable(false);
        jtxt_hmlp_epochs.setDisable(false);
        jtxt_hmlp_nbfacts.setDisable(false);
    }

    public void enable_nhf(){
        rd_nhf_woutpretrain.setDisable(false);
        rd_nhf_pretrain.setDisable(false);
        jtxt_nhf_epochs.setDisable(false);
        jtxt_nhf_batch.setDisable(false);
        jtxt_nhf_nbfacts.setDisable(false);
        jtxt_nhf_gmf_emb.setDisable(false);
        jtxt_nhf_hmlp_emb.setDisable(false);
        jtxt_nhf_layers.setDisable(false);
    }

    public void enable_mlp(){
        jtxt_mlp_batch.setDisable(false);
        jtxt_mlp_emb.setDisable(false);
        jtxt_mlp_epochs.setDisable(false);
        jtxt_mlp_layers.setDisable(false);
        jtxt_mlp_nbfacts.setDisable(false);
    }

    public void enable_ncf(){
        rd_ncf_pretrain.setDisable(false);
        rd_ncf_woutpretrain.setDisable(false);
        jtxt_ncf_nbfacts.setDisable(false);
        jtxt_ncf_epochs.setDisable(false);
        jtxt_ncf_gmf_emb.setDisable(false);
        jtxt_ncf_mlp_emb.setDisable(false);
        jtxt_ncf_layers.setDisable(false);
        jtxt_ncf_batch.setDisable(false);
    }

    //Select model events
    public void select_gmf(){
        empty_hmlp();
        empty_nhf();
        empty_mlp();
        empty_ncf();
        enable_gmf();
        disable_hmlp();
        disable_nhf();
        disable_mlp();
        disable_ncf();
    }

    public void select_hmlp(){
        empty_gmf();
        empty_nhf();
        empty_mlp();
        empty_ncf();
        enable_hmlp();
        disable_gmf();
        disable_nhf();
        disable_mlp();
        disable_ncf();
    }

    public void select_nhf(){
        empty_gmf();
        empty_hmlp();
        empty_mlp();
        empty_ncf();
        enable_nhf();
        disable_gmf();
        disable_hmlp();
        disable_mlp();
        disable_ncf();
    }

    public void select_mlp(){
        empty_ncf();
        empty_gmf();
        empty_hmlp();
        empty_nhf();
        enable_mlp();
        disable_ncf();
        disable_gmf();
        disable_hmlp();
        disable_nhf();
    }

    public void select_ncf(){
        empty_mlp();
        empty_gmf();
        empty_hmlp();
        empty_nhf();
        enable_ncf();
        disable_mlp();
        disable_gmf();
        disable_hmlp();
        disable_nhf();
    }

    //SLIDES METHODS
    private String imagesModels[] = {"GMF.png", "HybMLP.png", "NHybF.png"};
    private String titlesModels[] = {"GMF", "HybMLP", "NeuHybMF (NHybF)"};
    private String legendsModels[] = {"Generelized Matrix Factorization", "Hybrid Multilayer Perceptron", "Neural Hybrid Matrix Factorization"};
    private int index_img = 2;


    private String resultsHRML[] = {"HR hybmlp mlp GMF emb.PNG", "HR hybMLP MLP layers.PNG", "HR MLP HybMLP NHF NCF pf.PNG",
                                    "HR@K.PNG", "HR GMF MLP HybMLP NHF NCF Num_negs.PNG", "HR GMF MLP HybMLP NHF NCF new users Num_negs.PNG",
                                    "HR MLP HybMLP NHF NCF new users Num_negs.PNG", "HR@K new users with GMF.PNG", "HR@K new users without GMF.PNG"};

    private String resultsNDCGML[] = {"NDCG hybmlp mlp GMF emb.PNG", "NDCG hybMLP MLP layers.PNG", "NDCG MLP HybMLP NHF NCF pf.PNG",
                                      "NDCG@K.PNG", "NDCG GMF MLP HybMLP NHF NCF Num_negs.PNG", "NDCG GMF MLP HybMLP NHF NCF new users Num_negs.PNG",
                                      "NDCG MLP HybMLP NHF NCF new users Num_negs.PNG", "NDCG@K new users with GMF.PNG", "NDCG@K new users without GMF.PNG"};

    private String legendsHRML[] = {"HR with different embedding sizes", "HR with different number of layers", "HR with different predictive factors",
                                    "HR of Top K items' recommendation", "HR with different number of negatives", "HR with number of negatives and new users in the testset - with GMF",
                                    "HR different with number of negatives and new users in the testset - without GMF", "HR of Top K items' recommendation and new users in the testset - with GMF",
                                    "HR of Top K items' recommendation and new users in the testset - without GMF"};

    private String legendsNDCGML[] = {"NDCG with different embedding sizes", "NDCG with different number of layers", "NDCG with different predictive factors",
                                      "NDCG of Top K items' recommendation", "NDCG with different number of negatives", "NDCG with different number of negatives and new users in the testset - with GMF",
                                      "NDCG with different number of negatives and new users in the testset - without GMF", "NDCG of Top K items' recommendation and new users in the testset - with GMF",
                                      "NDCG@K new users - without GMF"};



    private String resultsHRY[] = {"HR hybmlp mlp GMF emb.PNG", "HR hybMLP MLP layers.PNG", "HR@K.PNG", "HR GMF MLP HybMLP NHF NCF Num_negs.PNG",
                                "HR GMF MLP HybMLP NHF NCF new users Num_negs.PNG", "HR@K new users with GMF.PNG"};

    private String resultsNDCGY[] = {"NDCG hybmlp mlp GMF emb.PNG", "NDCG hybMLP MLP layers.PNG", "NDCG@K.PNG", "NDCG GMF MLP HybMLP NHF NCF Num_negs.PNG",
                                    "NDCG GMF MLP HybMLP NHF NCF new users Num_negs.PNG", "NDCG@K new users with GMF.PNG"};

    private String legendsHRY[] = {"HR with different embedding sizes", "HR with different number of layers", "HR of Top K items' recommendation", "HR with different number of negatives",
                                    "HR with different number of negatives and new users in the testset", "HR of Top K items' recommendation and new users in the testset - with GMF"};

    private String legendsNDCGY[] = {"NDCG with different embedding sizes", "NDCG with different number of layers", "NDCG of Top K items' recommendation", "NDCG with different number of negatives",
                                    "NDCG with different number of negatives and new users in the testset", "NDCG of Top K items' recommendation and new users in the testset - with GMF"};


    private int index_results = 0;

    private String data = "ML1m";

    public int next(int index, String[] images){
        if(index<images.length-1)
            index++;
        else index=0;
        return index;
    }

    public int previous(int index, String[] images){
        if(index>0)
            index--;
        else index=images.length-1;
        return index;
    }

    public void nextImage(int index, String[] images1, String[] images2, String[] legends1, String[] legends2) throws URISyntaxException {
        index = next(index, images1);
        String path = getClass().getResource("/images/results/ML1m/" + images1[index]).toURI().toString();
        Image image = new Image(path);
        imgview_hr.setImage(image);
        area_hr.setPromptText(legends1[index]);
        path = getClass().getResource("/images/results/ML1m/" + images2[index]).toURI().toString();
        image = new Image(path);
        imgview_ndcg.setImage(image);
        area_ndcg.setPromptText(legends2[index]);
        index_results=index;
    }

    public void previousImage(int index, String[] images1, String[] images2, String[] legends1, String[] legends2) throws URISyntaxException {
        index = previous(index, images1);
        String path = getClass().getResource("/images/results/"+data+"/" + images1[index]).toURI().toString();
        Image image = new Image(path);
        imgview_hr.setImage(image);
        area_hr.setPromptText(legends1[index]);
        path = getClass().getResource("/images/results/"+data+"/" + images2[index]).toURI().toString();
        image = new Image(path);
        imgview_ndcg.setImage(image);
        area_ndcg.setPromptText(legends2[index]);
        index_results=index;
    }

    public void change_dataset() throws URISyntaxException {
        if(data.equals("ML1m")){
            label_dataset.setText("Yelp :");
            data=new String("Yelp");
            index_results=0;
            String path = getClass().getResource("/images/results/"+data+"/" + resultsHRY[index_results]).toURI().toString();
            Image image = new Image(path);
            imgview_hr.setImage(image);
            area_hr.setPromptText(legendsHRY[index_results]);
            path = getClass().getResource("/images/results/"+data+"/" + resultsNDCGY[index_results]).toURI().toString();
            image = new Image(path);
            imgview_ndcg.setImage(image);
            area_ndcg.setPromptText(legendsNDCGY[index_results]);
        }
        else {
            label_dataset.setText("MovieLens 1m :");
            data = new String("ML1m");
            index_results=0;
            String path = getClass().getResource("/images/results/"+data+"/" + resultsHRML[index_results]).toURI().toString();
            Image image = new Image(path);
            imgview_hr.setImage(image);
            area_hr.setPromptText(legendsHRML[index_results]);
            path = getClass().getResource("/images/results/"+data+"/" + resultsNDCGML[index_results]).toURI().toString();
            image = new Image(path);
            imgview_ndcg.setImage(image);
            area_ndcg.setPromptText(legendsNDCGML[index_results]);
        }
    }

    public void slide_next() throws URISyntaxException {
        if(data.equals("ML1m"))
            nextImage(index_results, resultsHRML, resultsNDCGML, legendsHRML, legendsNDCGML);
        else
            nextImage(index_results, resultsHRY, resultsNDCGY, legendsHRY, legendsNDCGY);
    }

    public void slide_previous() throws URISyntaxException {
        if(data.equals("ML1m"))
            previousImage(index_results, resultsHRML, resultsNDCGML, legendsHRML, legendsNDCGML);
        else
            previousImage(index_results, resultsHRY, resultsNDCGY, legendsHRY, legendsNDCGY);
    }


    private void nextImageModels(int index, String[] images, String[] legends, String[] titles) throws URISyntaxException {
        index = next(index, images);
        String path = getClass().getResource("/images/" + images[index]).toURI().toString();
        Image image = new Image(path);
        img_view.setImage(image);
        title.setText(titles[index]);
        area_models.setPromptText(legends[index]);
        index_img=index;
    }

    private void previousImageModels(int index, String[] images, String[] legends, String[] titles) throws URISyntaxException {
        index = previous(index, images);
        String path = getClass().getResource("/images/" + images[index]).toURI().toString();
        Image image = new Image(path);
        img_view.setImage(image);
        title.setText(titles[index]);
        area_models.setPromptText(legends[index]);
        index_img=index;
    }

    public void slideNext(MouseEvent mouseEvent) throws URISyntaxException {
        nextImageModels(index_img, imagesModels, legendsModels, titlesModels);
    }

    public void slidePrevious(MouseEvent mouseEvent) throws URISyntaxException {
        previousImageModels(index_img, imagesModels, legendsModels, titlesModels);
    }


    //DATASETS PANE METHODS
    private int selected_dataset(){
        if (rd_ml100k.isSelected())
            return 1;
        else
            if (rd_ml1m.isSelected())
                return 2;
            else
                if (rd_yelp.isSelected())
                    return 3;
                else return 0;
    }

    private int selected_split_type(){
        if(rd_tnormal.isSelected())
            return 1;
        else{
            if (rd_twnu.isSelected())
                return 2;
        }
        return 0;
    }

    private void select_path(){
        if (!jtxt_savepath.getText().isEmpty())
            path = jtxt_savepath.getText();
    }

    //BUTTON ACTIONS
    public void proceed_data() throws IOException {
        String cmd = new String("python C:/Users/ITEC/IdeaProjects/PFE_GUI/src/scripts/Dataset.py"); //PATH TO CHANGE
        int d = selected_dataset();
        int split = selected_split_type();
        select_path();

        //select a dataset
        switch(d){
            case 1 :
                cmd = cmd.concat(" --dataset 1");
                break;
            case 2 :
                cmd = cmd.concat(" --dataset 2");
                break;
            case 3 :
                cmd = cmd.concat(" --dataset 3");
                break;
        }

        //select how to split the dataset
        switch (split){
            case 1 :
                cmd = cmd.concat(" --type 2");
                break;
            case  2:
                cmd = cmd.concat(" --type 1");
                break;
        }

        //select number of negative instances per positive
        String negs = jtxt_numneg.getText();
        if(negs.matches("[0-9]+")) {
            switch (d){
                case 1 :
                    if (new Integer(negs) >= 14 || new Integer(negs)==0)
                        Warning_dialog("The value of number of negatives is too big", "An error may occur while transforming the dataset. Try numbers between 1 and 14.");
                    else {
                        cmd = cmd.concat(" --negs " + jtxt_numneg.getText());

                        //select path to save the train and test sets
                        if(path!=null)
                            cmd = cmd.concat(" --path_save " + path );

                        Runtime.getRuntime().exec("cmd /c start cmd.exe /K \"conda activate tfp36 && " + cmd + " && exit()\"\"");
                    }
                    break;
                case 2 :
                    if (new Integer(negs) >= 22 || new Integer(negs)==0)
                        Warning_dialog("The value of number of negatives is too big", "An error may occur while transforming the dataset. Try numbers between 1 and 22.");
                    else {
                        cmd = cmd.concat(" --negs " + jtxt_numneg.getText());

                        //select path to save the train and test sets
                        if(path!=null)
                            cmd = cmd.concat(" --path_save " + path);

                        Runtime.getRuntime().exec("cmd /c start cmd.exe /K \"conda activate tfp36 && " + cmd + " && exit()\"\"");
                    }
                    break;
                case 3 :
                    if (new Integer(negs) >= 231 || new Integer(negs)==0)
                        Warning_dialog("The value of number of negatives is too big", "An error may occur while transforming the dataset. Try numbers between 1 and 231.");
                    else {
                        cmd = cmd.concat(" --negs " + jtxt_numneg.getText());

                        //select path to save the train and test sets
                        if(path!=null)
                            cmd = cmd.concat(" --path_save " + path);

                        Runtime.getRuntime().exec("cmd /c start cmd.exe /K \"conda activate tfp36 && " + cmd + " && exit()\"\"");
                    }
                    break;
            }
        }
    }

    //Check estop >0
    public boolean check_estop(int patience){
        return patience > 1;
    }

    public void Warning_dialog(String header, String content){
        Alert alert = new Alert(Alert.AlertType.WARNING);
        alert.setTitle("WARNING");
        alert.setHeaderText(header);
        alert.setContentText(content);
        alert.showAndWait();
    }

    public void Error_dialog(String header, String content){
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle("ERROR");
        alert.setHeaderText(header);
        alert.setContentText(content);
        alert.showAndWait();
    }

    private int selected_model(){
        if(rd_gmf.isSelected())
            return 1;
        else{
            if(rd_hmlp.isSelected())
                return 2;
            else {
                if(rd_nhf.isSelected())
                    return 3;
                else {
                    if(rd_mlp.isSelected())
                        return 4;
                    else {
                        if (rd_ncf.isSelected())
                            return 5;
                    }
                }
            }
        }
        return 0;
    }

    public void start_training1() throws IOException{
        String cmd;

        select_path();
        if(path!=null)
            cmd = new String("python C:/Users/ITEC/IdeaProjects/PFE_GUI/src/scripts/Model.py --path_save "+path); //PATH TO CHANGE
        else
            cmd = new String("python C:/Users/ITEC/IdeaProjects/PFE_GUI/src/scripts/Model.py"); //PATH TO CHANGE

        int d = selected_dataset();
        int model = selected_model();

        if(d==0){
            Error_dialog("Dataset Not Found", "Select a dataset before training the model.");
        }
        else{
            switch (model) {
                case 1:
                    cmd = cmd.concat(" --model 1 --dataset "+d);
                    String gmf_emb = jtxt_gmf_emb.getText(), gmf_epochs = jtxt_gmf_epochs.getText(), gmf_batch = jtxt_gmf_batch.getText();
                    Object gmf_optimizer = cb_gmf_optimizer.getValue();

                    if(gmf_emb.matches("[0-9]+"))
                        cmd = cmd.concat(" --emb_size_gmf "+gmf_emb);

                    if(gmf_epochs.matches("[0-9]+"))
                        cmd = cmd.concat(" --epochs "+gmf_epochs);

                    if(gmf_batch.matches("[0-9]+"))
                        cmd = cmd.concat(" --batch_size "+gmf_batch);

                    if(gmf_optimizer!=null)
                        cmd = cmd.concat(" --optimizer_subs "+gmf_optimizer);
                    System.out.println(cmd);
                    break;
                case 2:
                    cmd = cmd.concat(" --model 2 --dataset "+ d);
                    String hmlp_emb = jtxt_hmlp_emb.getText(), hmlp_epochs = jtxt_hmlp_epochs.getText(), hmlp_batch = jtxt_hmlp_batch.getText(),
                            hmlp_layers = jtxt_hmlp_layers.getText(), hmlp_nbfacts = jtxt_hmlp_nbfacts.getText();
                    Object hmlp_optimizer = cb_hmlp_optimizer.getValue();

                    if(hmlp_emb.matches("[0-9]+"))
                        cmd = cmd.concat(" --emb_size_gmf "+hmlp_emb);

                    if(hmlp_epochs.matches("[0-9]+"))
                        cmd = cmd.concat(" --epochs "+hmlp_epochs);

                    if(hmlp_batch.matches("[0-9]+"))
                        cmd = cmd.concat(" --batch_size "+hmlp_batch);

                    if(hmlp_optimizer!=null)
                        cmd = cmd.concat(" --optimizer_subs "+hmlp_optimizer);

                    if(hmlp_nbfacts.matches("[0-9]+"))
                        cmd = cmd.concat(" --num_factors "+hmlp_nbfacts);

                    if(hmlp_layers.matches("[0-9]+"))
                        cmd = cmd.concat(" --num_layers "+hmlp_layers);
                    break;
                case 3:
                    cmd = cmd.concat(" --model 3 --dataset "+ d);
                    hmlp_emb = jtxt_nhf_hmlp_emb.getText();
                    gmf_emb = jtxt_nhf_gmf_emb.getText();
                    String nhf_epochs = jtxt_nhf_epochs.getText(), nhf_batch = jtxt_nhf_batch.getText(),
                           nhf_layers = jtxt_nhf_layers.getText(), nhf_nbfacts=jtxt_nhf_nbfacts.getText();
                    Object nhf_optimizer = cb_nhf_optimizer.getValue();
                    Object nhf_subs_optimizer = cb_nhf_submodels_optimizer.getValue();

                    if(hmlp_emb.matches("[0-9]+"))
                        cmd = cmd.concat(" --emb_size_hmlp "+hmlp_emb);

                    if(gmf_emb.matches("[0-9]+"))
                        cmd = cmd.concat(" --emb_size_gmf "+gmf_emb);

                    if(nhf_epochs.matches("[0-9]+"))
                        cmd = cmd.concat(" --epochs "+nhf_epochs);

                    if(nhf_batch.matches("[0-9]+"))
                        cmd = cmd.concat(" --batch_size "+nhf_batch);

                    if(nhf_optimizer!=null)
                        cmd = cmd.concat(" --optimizer_nhf "+nhf_optimizer);

                    if(nhf_subs_optimizer!=null)
                        cmd = cmd.concat(" --optimizer_subs "+nhf_subs_optimizer);

                    if(nhf_nbfacts.matches("[0-9]+"))
                        cmd = cmd.concat(" --num_factors "+ nhf_nbfacts);

                    if(nhf_layers.matches("[0-9]+"))
                        cmd = cmd.concat(" --num_layers "+nhf_layers);

                    if(rd_nhf_pretrain.isSelected())
                        cmd = cmd.concat(" --pretrain 1");
                    else{
                        if(rd_nhf_woutpretrain.isSelected())
                            cmd=cmd.concat(" --pretrain 0");
                    }
                    break;
            }
            if(model!=0) {
                if (check_estop.isSelected()) {
                    cmd = cmd.concat(" --estop 1");
                    String patience = jtxt_patience.getText();
                    if (patience.matches("[0-9]+")) {
                        if (!check_estop(new Integer(patience))) {
                            Error_dialog("Invalid number", "The number of useless epochs must be >= 1");
                        } else {
                            cmd = cmd.concat(" --estop " + patience);
                        }
                    }
                }
                Runtime.getRuntime().exec("cmd /c start cmd.exe /K \"conda activate tfp36 && " + cmd + " && exit()\"\"");
            }
        }
    }

    public void start_training2() throws IOException {
        String cmd;

        select_path();
        if(path!=null)
            cmd = new String("python C:/Users/ITEC/IdeaProjects/PFE_GUI/src/scripts/Model.py --path_save "+path); //PATH TO CHANGE
        else
            cmd = new String("python C:/Users/ITEC/IdeaProjects/PFE_GUI/src/scripts/Model.py"); //PATH TO CHANGE

        int d = selected_dataset();
        int model = selected_model();
        select_path();

        if(d==0){
            Error_dialog("Dataset Not Found", "Select a dataset before training the model.");
        }
        else {
            switch (model){
                case 4 :
                    cmd = cmd.concat(" --model 4 --dataset " + d);
                    String mlp_emb = jtxt_mlp_emb.getText(), mlp_epochs = jtxt_mlp_epochs.getText(), mlp_batch = jtxt_mlp_batch.getText(),
                            mlp_layers = jtxt_mlp_layers.getText(), mlp_nbfacts = jtxt_mlp_nbfacts.getText();
                    Object mlp_optimizer = cb_mlp_optimizer.getValue();

                    if (mlp_emb.matches("[0-9]+"))
                        cmd = cmd.concat(" --emb_size_gmf " + mlp_emb);

                    if (mlp_epochs.matches("[0-9]+"))
                        cmd = cmd.concat(" --epochs " + mlp_epochs);

                    if (mlp_batch.matches("[0-9]+"))
                        cmd = cmd.concat(" --batch_size " + mlp_batch);

                    if (mlp_optimizer != null)
                        cmd = cmd.concat(" --optimizer_subs " + mlp_optimizer);

                    if (mlp_nbfacts.matches("[0-9]+"))
                        cmd = cmd.concat(" --num_factors " + mlp_nbfacts);

                    if (mlp_layers.matches("[0-9]+"))
                        cmd = cmd.concat(" --num_layers " + mlp_layers);
                    System.out.println(cmd);
                    break;
                case 5 :
                    cmd = cmd.concat(" --model 5 --dataset " + d);
                    mlp_emb = jtxt_ncf_mlp_emb.getText();
                    String gmf_emb = jtxt_ncf_gmf_emb.getText(), ncf_epochs = jtxt_ncf_epochs.getText(),
                           ncf_batch = jtxt_ncf_batch.getText(), ncf_layers = jtxt_ncf_layers.getText(), ncf_nbfacts = jtxt_ncf_nbfacts.getText();
                    Object ncf_optimizer = cb_ncf_optimizer.getValue();
                    Object ncf_subs_optimizer = cb_ncf_submodels_optimizer.getValue();

                    if (mlp_emb.matches("[0-9]+"))
                        cmd = cmd.concat(" --emb_size_mlp " + mlp_emb);

                    if (gmf_emb.matches("[0-9]+"))
                        cmd = cmd.concat(" --emb_size_gmf " + gmf_emb);

                    if (ncf_epochs.matches("[0-9]+"))
                        cmd = cmd.concat(" --epochs " + ncf_epochs);

                    if (ncf_batch.matches("[0-9]+"))
                        cmd = cmd.concat(" --batch_size " + ncf_batch);

                    if (ncf_optimizer != null)
                        cmd = cmd.concat(" --optimizer_ncf " + ncf_optimizer);

                    if (ncf_subs_optimizer != null)
                        cmd = cmd.concat(" --optimizer_subs " + ncf_subs_optimizer);

                    if (ncf_nbfacts.matches("[0-9]+"))
                        cmd = cmd.concat(" --num_factors " + ncf_nbfacts);

                    if (ncf_layers.matches("[0-9]+"))
                        cmd = cmd.concat(" --num_layers " + ncf_layers);

                    if (rd_ncf_pretrain.isSelected())
                        cmd = cmd.concat(" --pretrain 1");
                    else {
                        if (rd_ncf_woutpretrain.isSelected())
                            cmd = cmd.concat(" --pretrain 0");
                    }
                    break;
            }
            if(model!=0){
                if (check_estop2.isSelected()) {
                    cmd = cmd.concat(" --estop 1");
                    String patience = jtxt_patience2.getText();
                    if(patience.matches("[0-9]+")){
                        if(!check_estop(new Integer(patience)))
                            Error_dialog("Invalid number","The number of useless epochs must be >= 1");
                        else
                            cmd = cmd.concat(" --estop "+patience);
                    }
                }
                Runtime.getRuntime().exec("cmd /c start cmd.exe /K \"conda activate tfp36 && " + cmd + " && exit()\"\"");
            }
        }
    }

}

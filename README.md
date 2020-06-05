SETUP

1.
pip install -r requirements.txt

2.
you must build somethings from CTPN

nagivate to project directory root
cd ./Code/ctpn/utils/bbox/ && ./make.sh

3.
copy ctpn model 

download from https://drive.google.com/file/d/1HcZuB_MHqsKhKEKpfF1pEU85CYy4OlWO/view?usp=sharing


extract the .zip to  ./Code/

mv checkpoint_mlt/* ./Checkpoints_ctpn/

rmdir checkpoint_mlt

 


TODO
- make this readme pretty
- complete modules

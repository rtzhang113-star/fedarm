WS-DREAM Dataset 3: time-aware Web service QoS

Obtain the dataset from the provider and place rtRate and tpRate in this
folder. Raw data is not distributed in this code package.

Sources:
https://www.inpluslab.com/wsdream/
https://inpluslab.com/wsdream/download/WSDream-QoSDataset3.zip
https://github.com/LanternYing/Dataset

The author-associated GitHub archive provides the dataset description;
the large Dataset 3 files may need to be obtained from the official source.
Follow the data provider's research-use and redistribution terms.

Reference:
Yilei Zhang, Zibin Zheng, and Michael R. Lyu.
WSPred: A Time-Aware Personalized QoS Prediction Framework for Web Services.
IEEE International Symposium on Software Reliability Engineering, 2011,
pp. 210-219. DOI: 10.1109/ISSRE.2011.17

This implementation uses arrays with shape (64 time slices, 142 users,
4500 services). The loader expects each raw line to contain:
    user_id service_id time_id value
with zero-based IDs, matching continual_qos/data.py. RT is in seconds;
TP is in kbps. Check the file format before converting a different export.

RT validity: finite values strictly between 0 and 19.9.
TP validity: finite positive values.
RT duplicate policy: any-valid-record-wins.
TP duplicate policy: last-record-wins.
Invalid or absent entries are represented as NaN in generated caches.
Caches are generated locally under data/cache and are not part of the release.

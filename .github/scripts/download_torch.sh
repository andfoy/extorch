set -exou

BASE_URL="https://download.pytorch.org/libtorch"

if [[ $PYTORCH_VERSION == "latest" ]]; then
    BASE_URL="${BASE_URL}/nightly"
fi

FILENAME="libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}.zip"
BASE_URL="${BASE_URL}/${DEVICE}/${FILENAME}"

pushd native/extorch_native
wget $BASE_URL
unzip $FILENAME
rm -rf $FILENAME
popd

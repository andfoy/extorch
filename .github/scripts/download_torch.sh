set -exou

BASE_URL="https://download.pytorch.org/libtorch"

FILENAME="libtorch-shared-with-deps-${PYTORCH_VERSION}.zip"
ZIP_FILENAME=$FILENAME

if [[ $PYTORCH_VERSION == "latest" ]]; then
    BASE_URL="${BASE_URL}/nightly"
else
    if [[ $DEVICE == "cpu" ]]; then
        FILENAME="libtorch-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip"
        ZIP_FILENAME="libtorch-shared-with-deps-${PYTORCH_VERSION}+cpu.zip"
    fi
fi

BASE_URL="${BASE_URL}/${DEVICE}/${FILENAME}"

pushd priv/native
wget $BASE_URL -q --show-progress
unzip $ZIP_FILENAME
rm -rf $ZIP_FILENAME
popd

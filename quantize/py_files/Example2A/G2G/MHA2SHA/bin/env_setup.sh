# !/bin/bash
# ----------------
#
# Qualcomm Technologies, Inc. Proprietary
# (c) 2022-24 Qualcomm Technologies, Inc. All rights reserved.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
#
# -----------------------------------------------------------------------------

# This script is inspired from the QAIRT env_setup.sh

function usage()
{
  clean_up_error;
  cat << EOF
Script sets up environment variables for the MHA2SHA tool.

USAGE: 
  $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-h]

EOF
}

function clean_up_error()
{
  if [[ -n "$OLD_PYTHONPATH"  ]]
  then
    export PYTHONPATH=$OLD_PYTHONPATH
  fi
}


OLD_PYTHONPATH=$PYTHONPATH

if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  echo "[ERROR] This file should be run with 'source'"
  usage;
  return 1;
fi

OPTIND=1
while getopts "h?s:m:" opt; do
    case "$opt" in
    h)
        usage;
        return 0
        ;;
    \?)
    usage;
    return 1 ;;
    esac
done


# Get the source dir of the env_setup.sh script
SOURCEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
MHA2SHA_ROOT=$(readlink -f ${SOURCEDIR}/..)

export PYTHONPATH=$PYTHONPATH:$MHA2SHA_ROOT/src/python
export PATH=$MHA2SHA_ROOT/bin:$PATH

echo "MHA2SHA tool root set to:- "$MHA2SHA_ROOT

python_version=$(python --version)
echo "Python Version:- $python_version"

# Clean up the local variables
unset mha2sha_root python_version OLD_PYTHONPATH SOURCEDIR MHA2SHA_ROOT

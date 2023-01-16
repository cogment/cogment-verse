// Copyright 2022 AI Redefined Inc. <dev+cogment@ai-r.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import ndarray_pb from "../ndarray_pb";
import spaces_pb from "../spaces_pb";
import data_pb from "../data_pb";

const NdArray = ndarray_pb.cogment_verse.nd_array.Array;
export const DType = ndarray_pb.cogment_verse.nd_array.DType;
export const Space = spaces_pb.cogment_verse.spaces.Space;
const PlayerAction = data_pb.cogment_verse.PlayerAction;
const TeacherAction = data_pb.cogment_verse.TeacherAction;

export const TEACHER_NOOP_ACTION = new TeacherAction({});

const serializeNdArray = (dtype, shape, flattenedArray) => {
  let arrayValue;
  switch (dtype) {
    case DType.DTYPE_INT8:
      arrayValue = new Int8Array(flattenedArray);
      break;
    case DType.DTYPE_FLOAT32:
      arrayValue = new Float32Array(flattenedArray);
      break;
    default:
      throw new Error(`Unsupported dtype [${dtype}]`);
  }
  const bytesValue = new Uint8Array(arrayValue.buffer);
  return new NdArray({
    dtype: dtype,
    shape: shape,
    rawData: bytesValue,
  });
};

const deserializeNdArray = (serializedArray) => {
  const dtype = serializedArray.dtype;
  const shape = serializedArray.shape;
  const length = shape.reduce((accFlatdim, dim) => dim * accFlatdim, 1);
  let typedArray;
  switch (dtype) {
    case DType.DTYPE_INT8:
      typedArray = new Int8Array(serializedArray.rawData.buffer, serializedArray.rawData.byteOffset, length);
      break;
    case DType.DTYPE_FLOAT32:
      typedArray = new Float32Array(serializedArray.rawData.buffer, serializedArray.rawData.byteOffset, length);
      break;
    default:
      throw new Error(`Unsupported dtype [${dtype}]`);
  }
  return [dtype, shape, typedArray];
};

const flatdim = (space) => {
  if (space.discrete) {
    return space.discrete.n;
  }
  if (space.box) {
    const shape = space.box.low.shape;
    return shape.reduce((accFlatdim, dim) => dim * accFlatdim, 1);
  }
  if (space.dict) {
    return Object.values(space.dict.spaces).reduce((accFlatdim, subSpace) => accFlatdim + flatdim(subSpace.space), 0);
  }
  throw new Error(`Unsupported space [${JSON.stringify(space)}]`);
};

const flattype = (space) => {
  if (space.discrete) {
    return DType.DTYPE_INT8;
  }
  if (space.box) {
    return space.box.low.dtype;
  }
  if (space.dict) {
    // TODO look into that
    return DType.DTYPE_FLOAT64;
  }
  throw new Error(`Unsupported space [${JSON.stringify(space)}]`);
};

const deepFlattenArray = (array) =>
  array.reduce((flattenedArray, value) => {
    if (Array.isArray(value)) {
      flattenedArray.concat(deepFlattenArray(value));
    } else {
      flattenedArray.push(value);
    }
    return flattenedArray;
  }, []);

const flatten = (space, value) => {
  if (space.discrete) {
    const flattenedValue = new Array(space.discrete.n);
    const valueIndex = space.discrete.start ? value - space.discrete.start : value;
    flattenedValue.fill(0);
    flattenedValue[valueIndex] = 1;
    return flattenedValue;
  }
  if (space.box) {
    return deepFlattenArray(value);
  }
  if (space.dict) {
    const orderedValues = space.dict.spaces.map((subSpace) => flatten(subSpace.space, value[subSpace.key]));
    return deepFlattenArray(orderedValues);
  }
  throw new Error(`Unsupported space [${JSON.stringify(space)}]`);
};

const deepUnflattenArray = (shape, array) => {
  if (shape.length < 1) {
    throw new Error(`Shape need to have at least one item`);
  }
  if (shape.length === 1) {
    if (shape[0] !== array.length) {
      throw new Error(`Array [${array}] is of unexpected length [${array.length}], expected [${shape[0]}]`);
    }
    return array;
  }
  const targetLastSize = shape[shape.length - 1];
  const unflattenedArray = array.reduce(
    (unflattenedValue, item) => {
      const lastSize = unflattenedValue[unflattenedValue.length - 1].length;
      if (lastSize === targetLastSize) {
        unflattenedValue.push([item]);
      } else {
        unflattenedValue[unflattenedValue.length - 1].push(item);
      }
      return unflattenedValue;
    },
    [[]]
  );
  return deepUnflattenArray(shape.slice(0, shape.length - 1), unflattenedArray);
};

const unflatten = (space, flattenedValue) => {
  if (space.discrete) {
    return flattenedValue.findIndex((v) => v !== 0) + (space.discrete.start || 0);
  }
  if (space.box) {
    return deepUnflattenArray(space.box.low.shape, flattenedValue);
  }
  if (space.dict) {
    const { value, currentIndex } = space.dict.spaces.reduce(
      ({ value, currentIndex }, subSpace) => {
        const subSpaceFlatDim = flatdim(subSpace.space);
        const nextIndex = currentIndex + subSpaceFlatDim;
        const subSpaceValue = flattenedValue.slice(currentIndex, nextIndex);
        value[subSpace.key] = unflatten(subSpace.space, subSpaceValue);
        return { value, currentIndex: nextIndex };
      },
      { value: {}, currentIndex: 0 }
    );
    if (currentIndex !== flattenedValue.length) {
      throw new Error(
        `Unflatten operation finished before index [${currentIndex}] while the flattened value has [${flattenedValue.length}] items`
      );
    }
    return value;
  }
  throw new Error(`Unsupported space [${JSON.stringify(space)}]`);
};

// Discrete value => Number, e.g. 4
// Box value => Array of numbers, e.g [4.4, 2, -1.03]
// Dict action => Dict of actions, e.g {"action_1": 1, "action_2": -3}
const serializeSpaceValue = (space, value) => {
  return serializeNdArray(flattype(space), [flatdim(space)], flatten(space, value));
};

const deserializeSpaceValue = (space, serializedValue) => {
  const [, shape, flattenedValue] = deserializeNdArray(serializedValue);
  // TODO look into dtype checking
  // const [dtype, shape, flattenedValue] = deserializeNdArray(serializedValue);
  // if (dtype !== flattype(space)) {
  //   throw new Error(`Unexpected value type [${dtype}] for space [${JSON.stringify(space)}]`);
  // }
  if (shape.length !== 1 || shape[0] !== flatdim(space)) {
    throw new Error(
      `Unexpected value shape [${JSON.stringify(shape)}] for space [${JSON.stringify(
        space
      )}], expected [${JSON.stringify([flatdim(space)])}]`
    );
  }
  return unflatten(space, flattenedValue);
};

export const serializePlayerAction = (space, value) =>
  new PlayerAction({
    value: serializeSpaceValue(space, value),
  });

export const deserializeObservationValue = (space, observation) => {
  const observationValue = observation?.value;
  if (observationValue == null) {
    return null;
  }
  return deserializeSpaceValue(space, observationValue);
};

export const deserializeObservationActionMask = (space, observation) => {
  const observationActionMask = observation?.actionMask;
  if (observationActionMask == null) {
    return null;
  }
  return deserializeSpaceValue(space, observationActionMask);
};

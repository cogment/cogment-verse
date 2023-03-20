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

import classNames from "classnames";
import { useCallback, useMemo, useState } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCaretLeft, faCaretDown } from "@fortawesome/free-solid-svg-icons";

import {
  deserializeObservationActionMask,
  deserializeObservationSpace,
  deserializeObservationValue,
} from "../utils/spaceSerialization";

const SingleValue = ({ value }) => <span>{value}</span>;

const VectorValue = ({ value }) => <span>{JSON.stringify(value).replaceAll(",", ", ")}</span>;

const TensorValue = ({ value, shape }) => {
  if (shape.length <= 1) {
    return <VectorValue value={value} />;
  } else {
    return (
      <>
        <span>[</span>
        <table className="table-fixed ml-2">
          <tbody>
            {value.map((row, rowIndex) => {
              return (
                <tr key={rowIndex}>
                  {row.map((cell, cellIndex) => (
                    <td key={cellIndex}>
                      {cellIndex === 0 ? <span>[</span> : null}
                      <VectorValue value={cell} />
                      {cellIndex === row.length - 1 ? <span>{"], "}</span> : <span>{", "}</span>}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
        <span>]</span>
      </>
    );
  }
};

const SpaceValue = ({ value, space, className, ...props }) => (
  
  <div className={classNames(className, "block font-mono")} {...props}>
    {space.discrete ? (
      <>
        <h3 className={classNames(className, "font-mono")} {...props}>
            type: {space.kind}(n={space.discrete.n}, start={space.discrete.start})
        </h3>
        value: <SingleValue value={value} />
      </>
    ) : space.box ? (
      <>
        <h3 className={classNames(className, "font-mono")} {...props}>
            type: {space.kind}(low.shape={space.box.low.shape}, high.shape={space.box.high.shape})
        </h3>
        value: <TensorValue value={value} shape={space.box.low.shape} />
      </>
    ) : space.multiBinary ? (
      <>
        <h3 className={classNames(className, "font-mono")} {...props}>
        type: {space.kind}(n={space.multiBinary.n.int32Data})
        </h3>
        value: <TensorValue value={value} shape={space.multiBinary.n.int32Data} />
      </>
    ) : space.multiDiscrete ? (
      <>
        <h3 className={classNames(className, "font-mono")} {...props}>
        type: {space.kind}(nvec={space.multiDiscrete.nvec.int32Data})
        </h3>
        value: <TensorValue value={value} shape={space.multiDiscrete.nvec.shape} />
      </>
    ) : space.dict ? (
      <div>
        {
          space.dict.spaces.map( ({key, space}) => (
              <>
                <h3 className={classNames(className, "font-semibold lowercase mt-1")} {...props}>
                  {key}:
                </h3>
                <SpaceValue value={value[key]} space={space} />
              </>
          )
          )
        }
      </div>
    ) : (
      `Unsupported space [${JSON.stringify(space, null, `\t`)}]`
    )}
  </div>
);

const AttributeTitle = ({ className, children, ...props }) => (
  <h3 className={classNames(className, "font-semibold bg-indigo-200 lowercase mt-1")} {...props}>
    {children}
  </h3>
);

const AttributeValue = ({ className, children, ...props }) => (
  <span className={classNames(className, "font-mono")} {...props}>
    {children}
  </span>
);

const ObservationInspector = ({ event, actorParams }) => {
  const serializedObservationSpace = actorParams?.config?.environmentSpecs?.observationSpace;
  const { observationSpace, actionMaskSpace } = useMemo(
    () => deserializeObservationSpace(serializedObservationSpace),
    [serializedObservationSpace]
  );
  if (!observationSpace || !event.observation) {
    return null;
  }
  const value = deserializeObservationValue(observationSpace, event.observation);
  const actionMask = deserializeObservationActionMask(actionMaskSpace, event.observation);
  const currentPlayer = event.observation.currentPlayer;
  return (
    <>
      {currentPlayer != null ? (
        <>
          <AttributeTitle>Current Player</AttributeTitle>
          <AttributeValue>{event.observation.currentPlayer}</AttributeValue>
        </>
      ) : null}
      <AttributeTitle>Observation</AttributeTitle>
      <SpaceValue value={value} space={observationSpace} />
      {actionMask != null ? (
        <>
          <AttributeTitle>Action Mask</AttributeTitle>
          <SpaceValue value={actionMask} space={actionMaskSpace} />
        </>
      ) : null}
    </>
  );
};

export const Inspector = ({ trialId, event, actorParams, className, ...props }) => {
  const [visible, setVisible] = useState(false);
  const toggleVisible = useCallback(() => setVisible((visible) => !visible), [setVisible]);

  const environment = actorParams?.config?.environmentSpecs?.implementation || undefined;
  const runId = actorParams?.config?.runId;

  return (
    <div className={classNames(className, "flex flex-col items-stretch relative")} {...props}>
      <button className={classNames("flex flex-row justify-between items-center p-1")} onClick={toggleVisible}>
        <h1 className="font-mono text-lg font-semibold">
          {runId}/{trialId}
        </h1>
        <FontAwesomeIcon icon={visible ? faCaretDown : faCaretLeft} />
      </button>
      {visible ? (
        <div
          className={classNames(
            className,
            "flex flex-col items-start absolute p-2 bg-neutral-400/90 rounded-b z-10 top-full inset-x-0"
          )}
          {...props}
        >
          <AttributeTitle>Environment</AttributeTitle>
          <AttributeValue>{environment}</AttributeValue>
          <ObservationInspector event={event} actorParams={actorParams} />
        </div>
      ) : null}
    </div>
  );
};

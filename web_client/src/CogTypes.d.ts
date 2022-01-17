




/*
 *  Copyright 2021 AI Redefined Inc. <dev+cogment@ai-r.com>
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WIanyHOUany WARRANanyIES OR CONDIanyIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

import { Constructor, IConversionOptions, Properties, Reader, Writer } from "protobufjs";

export class Message {

    /**
     * Constructs a new message instance.
     * @param [properties] Properties to set
     */
    constructor(properties?: Properties<any>)

    /**
     * Creates a new message of this type using the specified properties.
     * @param [properties] Properties to set
     * @returns Message instance
     */
    public static create(this: Constructor<Message>, properties?: { [k: string]: any }): Message

    /**
     * Encodes a message of this type.
     * @param message Message to encode
     * @param [writer] Writer to use
     * @returns Writer
     */
    public static encode(this: Constructor<Message>, message: (any | { [k: string]: any }), writer?: Writer): Writer;

    /**
     * Encodes a message of this type preceeded by its length as a varint.
     * @param message Message to encode
     * @param [writer] Writer to use
     * @returns Writer
     */
    public static encodeDelimited(this: Constructor<Message>, message: (any | { [k: string]: any }), writer?: Writer): Writer;

    /**
     * Decodes a message of this type.
     * @param reader Reader or buffer to decode
     * @returns Decoded message
     */
    public static decode(this: Constructor<Message>, reader: (Reader | Uint8Array)): Message;

    /**
     * Decodes a message of this type preceeded by its length as a varint.
     * @param reader Reader or buffer to decode
     * @returns Decoded message
     */
    public static decodeDelimited(this: Constructor<Message>, reader: (Reader | Uint8Array)): Message;

    /**
     * Verifies a message of this type.
     * @param message Plain object to verify
     * @returns 'null' if valid, otherwise the reason why it is not
     */
    public static verify(message: { [k: string]: any }): (string | null);

    /**
     * Creates a new message of this type from a plain object. Also converts values to their respective internal types.
     * @param object Plain object
     * @returns Message instance
     */
    public static fromObject(this: Constructor<Message>, object: { [k: string]: any }): Message;

    /**
     * Creates a plain object from a message of this type. Also converts values to other types if specified.
     * @param message Message instance
     * @param [options] Conversion options
     * @returns Plain object
     */
    public static toObject(this: Constructor<Message>, message: Message, options?: IConversionOptions): { [k: string]: any };

    /**
     * Converts this message to JSON.
     * @returns JSON object
     */
    public toJSON(): { [k: string]: any };
}

export type MessageBase = Message;

export type MessageClass = typeof Message


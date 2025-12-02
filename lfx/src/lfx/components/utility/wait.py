import asyncio

from lfx.custom.custom_component.component import Component
from lfx.io import DataInput, IntInput, Output
from lfx.schema.data import Data
from lfx.schema.message import Message


class WaitComponent(Component):
    display_name = "Wait"
    description = "Pause the workflow for a specified number of seconds."
    icon = "Timer"
    name = "Wait"

    inputs = [
        IntInput(
            name="duration",
            display_name="Duration (seconds)",
            value=1,
            info="Number of seconds to wait.",
            min=0,
        ),
        DataInput(
            name="trigger",
            display_name="Trigger",
            info="Optional input used to start the wait. The payload is forwarded after the delay.",
            required=False,
        ),
    ]

    outputs = [
        Output(
            name="wait_done",
            display_name="Wait Complete (Message)",
            method="wait_message",
        ),
        Output(
            name="wait_data",
            display_name="Wait Complete (Data)",
            method="wait_data",
        ),
    ]

    async def _wait(self) -> None:
        duration = max(0, int(getattr(self, "duration", 0)))
        if duration > 0:
            await asyncio.sleep(duration)

    def _trigger_payload(self):
        payload = getattr(self, "trigger", None)
        if isinstance(payload, (Message, Data)):
            return payload
        if payload is not None:
            return Data(data={"value": payload})
        return Message()

    async def wait_message(self) -> Message:
        await self._wait()
        trigger_payload = self._trigger_payload()
        if isinstance(trigger_payload, Message):
            return trigger_payload
        if isinstance(trigger_payload, Data):
            return Data(
                data=trigger_payload.data,
                text=trigger_payload.text,
                text_key=trigger_payload.text_key,
                default_value=trigger_payload.default_value,
            ).to_message()
        return Message()

    async def wait_data(self) -> Data:
        await self._wait()
        trigger_payload = self._trigger_payload()
        if isinstance(trigger_payload, Data):
            return trigger_payload
        if isinstance(trigger_payload, Message):
            return Data(text=trigger_payload.text)
        return Data(data={})



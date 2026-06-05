import logging

from trackey.core.factories.builder import Builder
from trackey.core.events.bus import EventBus
from trackey.core.registries.subscriber import SUBSCRIBER_REGISTRY
from trackey.core.registries.event import EVENT_REGISTRY
from trackey.core.events.throttle import ThrottledSubscriber

logger = logging.getLogger(__name__)


class EventBusBuilder(Builder):

    SUB_CFG_FORMAT = (
        "subscribers:\n"
        "  - type: <subscriber-type>\n"
        "    on: [event_name_1, event_name_2]\n"
        "    params:\n"
        "      <subscriber-specific-params>\n"
        "\n"
        "mail params:          to: [mail1, mail2]\n"
        "image_storage params: path: <path-to-save-snapshots>\n"
        "webhook params:       url: <webhook-url>\n"
    )

    def __init__(self, cfg_path: str):
        self.cfg = self._load_yaml(cfg_path)

    def build(self) -> EventBus:
        event_bus = EventBus()
        self._build_bus(event_bus)
        return event_bus

    # ------------------------------------------------------------------ #
    # Build                                                              #
    # ------------------------------------------------------------------ #

    def _build_bus(self, event_bus: EventBus) -> None:
        subscribers_cfg = self.cfg.get("subscribers", [])

        for sub_cfg in subscribers_cfg:
            self._validate_sub(sub_cfg)

            subscriber = self._build_subscriber(sub_cfg)

            # wire subscriber to event types from config
            for event_name in sub_cfg["listen"]:
                event_type = EVENT_REGISTRY.get(event_name)
                if not event_type:
                    logger.error(
                        f"[EventBusBuilder] Unknown event: '{event_name}'. "
                        f"Available: {list(EVENT_REGISTRY.keys())}"
                    )
                    raise ValueError(
                        f"[EventBusBuilder] Unknown event: '{event_name}'. "
                        f"Available: {list(EVENT_REGISTRY.keys())}"
                    )
                event_bus.subscribe(event_type, subscriber)
                logger.info(
                    f"[EventBusBuilder] {sub_cfg['type']} "
                    f"subscribed to '{event_name}'"
                )

    def _build_subscriber(self, sub_cfg: dict) -> ThrottledSubscriber:
        sub_type         = sub_cfg["type"]
        throttle_seconds = sub_cfg.get("throttle_seconds", 0.0)

        plugin_cls = SUBSCRIBER_REGISTRY.get(sub_type)
        if not plugin_cls:
            raise ValueError(
                f"[EventBusBuilder] Unknown subscriber type: '{sub_type}'. "
                f"Available: {list(SUBSCRIBER_REGISTRY.keys())}"
            )

        plugin_cls.validate(sub_cfg)
        subscriber = plugin_cls.build(sub_cfg)

        # always wrap — ThrottledSubscriber with 0.0 is transparent
        return ThrottledSubscriber(subscriber, throttle_seconds)

    # ------------------------------------------------------------------ #
    # Validation                                                         #
    # ------------------------------------------------------------------ #

    def _validate_sub(self, sub_cfg: dict) -> None:
        self._validate_required_fields(sub_cfg)
        self._validate_sub_type(sub_cfg)
        self._validate_events(sub_cfg)


    def _validate_required_fields(self, sub_cfg: dict) -> None:
        if "type" not in sub_cfg:
            raise ValueError(
                f"[EventBusBuilder] Subscriber missing 'type'.\n"
                f"{self.SUB_CFG_FORMAT}"
            )
        if "listen" not in sub_cfg:
            raise ValueError(
                f"[EventBusBuilder] Subscriber missing 'on' "
                f"(list of event names).\n"
                f"{self.SUB_CFG_FORMAT}"
            )

    def _validate_sub_type(self, sub_cfg: dict) -> None:
        sub_type = sub_cfg["type"]
        if sub_type not in SUBSCRIBER_REGISTRY:
            raise ValueError(
                f"[EventBusBuilder] Unknown subscriber type: '{sub_type}'. "
                f"Available: {list(SUBSCRIBER_REGISTRY.keys())}"
            )

    def _validate_events(self, sub_cfg: dict) -> None:
        on_events = sub_cfg["listen"]
        if not isinstance(on_events, list):
            raise ValueError(
                f"[EventBusBuilder] 'on' must be a list of event names.\n"
                f"{self.SUB_CFG_FORMAT}"
            )
        if not on_events:
            raise ValueError(
                f"[EventBusBuilder] 'on' must contain at least one event.\n"
                f"{self.SUB_CFG_FORMAT}"
            )

from sycamore.transforms.property_extraction.types import RichProperty, AttributionValue
from sycamore.transforms.property_extraction.attribution import TextMatchAttributionStrategy
from sycamore.data import Document, Element
from sycamore.utils.zip_traverse import zip_traverse


def test_refine_attribution():
    strategy = TextMatchAttributionStrategy()

    element_texts = [
        "Extreme caution required Subject Page ref. As long as the airbag is activated, persons with disability are advised not to travel in the Child safety Page"
        " 40 vehicle in order to avoid the risk of serious injuries or death, even in minor crashes. Transport of person with disability The airbag is not a"
        " substitute for the seat belts. Correct use of the seat belts, in combination Airbag Page 41 with the airbag, will offer protection for the driver and "
        "passenger in the front seats in the event of a head-on collision. Front airbags cannot offer protection in side crashes, certain front-angular crashes, "
        "roll Airbag Page 41 over events or in secondary impacts (if a second crash happens after the airbags have been deployed in a previous crash). The seat "
        "belts are designed to help reduce the risk of injuries in roll over events and secondary front impacts. A properly fastened seat belt is needed to help "
        "protect occupants in roll over events and secondary front impacts. Front airbags are designed to not be deployed in low severity frontal crashes. The "
        "seat belts Airbag Page 41 can help reduce injuries in low severity crashes. A properly fastened seat belt is needed to help protect the occupants in low "
        "severity frontal crashes. When the ignition key is turned to position II, the warning light C illuminates. If no Airbag Page 41 malfunctioning is detected, "
        "it will go off after 4 seconds. If the warning light does not Airbag system components illuminate, if it remains on or if it illuminates while driving, "
        "contact your Authorized Ferrari Dealer immediately. The driver and passenger are both advised not to travel handling objects (e.g., beverage cans Airbag "
        "Page 42 or bottles, pipes, etc.) that could cause injury in the case of airbag deployment. Operation Always drive with your hands on the rim of the "
        "steering wheel so that, in case of activation, Airbag Page 42 the airbag can deploy without obstruction. Driving with your hands inside the steering Operation "
        "wheel rim or on the airbag cover increases the risk of injury for your wrists and arms. The driver and passenger must always fasten their seat belts and sit "
        "in an upright position, as Airbag Page 42 far as possible away from the airbag, in order to help ensure protection in all types of collision. Operation",
        "Something not quite that long about the value of colorful hydrangeas",
        "95,000",
    ]

    propsdict = {
        "a": "colorful hydrangeas",
        "b": 95000,
        "c": {
            "d": "colorful hydrangeas",
            "e": "Seat belts!",
            "f": ["Child safety", "ignition key to position 2", 9500],
        },
    }

    doc = Document(
        elements=[
            Element(text_representation=t, properties={"page_number": i, "_element_index": i * 5})
            for i, t in enumerate(element_texts)
        ]
    )

    rp = RichProperty.from_prediction(propsdict)
    for k, (v,), (p,) in zip_traverse(rp):
        v.attribution = AttributionValue(
            element_indices=[e.element_index if e.element_index is not None else -1 for e in doc.elements]
        )
    richprops = rp.value

    atta = strategy.refine_attribution(richprops["a"], doc).attribution
    assert atta is not None
    assert atta.element_indices == [5]
    assert atta.page == 1
    assert atta.text_snippet == "colorful hydrangeas"
    assert atta.text_match_score == 1.0
    assert atta.text_span == (49, 68)

    attb = strategy.refine_attribution(richprops["b"], doc).attribution
    assert attb is not None
    assert attb.element_indices == [10]
    assert attb.page == 2
    assert attb.text_snippet == "95,000"
    assert attb.text_match_score is not None
    assert attb.text_match_score < 1.0
    assert attb.text_span == (0, 7)

    pc = strategy.refine_attribution(richprops["c"], doc)
    assert pc.attribution is None

    pd = pc.value["d"]
    attd = pd.attribution
    assert attd is not None
    assert attd.element_indices == [5]
    assert attd.page == 1
    assert attd.text_snippet == "colorful hydrangeas"
    assert attd.text_match_score == 1.0
    assert attd.text_span == (49, 68)

    pe = pc.value["e"]
    atte = pe.attribution
    assert atte is not None
    assert atte.element_indices == [0]
    assert atte.page == 0
    assert atte.text_snippet == " seat belts"
    assert atte.text_match_score < 1.0

    pf = pc.value["f"]
    assert pf.attribution is None

    pf0 = pf.value[0]
    attf0 = pf0.attribution
    assert attf0 is not None
    assert attf0.element_indices == [0]
    assert attf0.page == 0
    assert attf0.text_snippet == "Child safety"
    assert attf0.text_match_score == 1.0

    pf1 = pf.value[1]
    attf1 = pf1.attribution
    assert attf1 is not None
    assert attf1.element_indices == [0]
    assert attf1.page == 0
    assert attf1.text_snippet == "e ignition key is turned to position "
    assert attf1.text_match_score < 1.0

    pf2 = pf.value[2]
    attf2 = pf2.attribution
    assert attf2 is not None
    assert attf2.element_indices == [10]
    assert attf2.page == 2
    assert attf2.text_snippet == "95,000"
    assert attf2.text_match_score is not None
    assert attf2.text_match_score < 1.0
    assert attf2.text_span == (0, 6)

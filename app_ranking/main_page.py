from typing import Callable
import solara
from state import StateController, State, StateErrorType

controller = StateController()

style_title = "text-align: center;margin-top: 50px; font-size: 2rem; font-weight: bold;"
style_row = "display: flex; width: 100%; justify-content: space-around;"
style_col = "display: flex; flex-direction: column; align-items: center;height: 100%; width: 50%; background-color: #f5f5f5; border-radius: 10px; padding: 10px !important; transition: all 1s ease-in-out; justify-content: space-between;"
style_input = "background-color: #f5f5f5; border-radius: 10px; padding: 10px 20px 5px 20px; transition: all 1s ease-in-out; height: fit-content; width: 40%;"
style_err_label = "color: red; font-size: 12px; margin-bottom: 5px; margin-top: -10px; line-height: 12px; word-break: break-word;overflow-wrap: break-word;word-wrap: break-word;-webkit-hyphens: auto;-ms-hyphens: auto;hyphens: auto;"


@solara.component
def Page():
    state, set_state = solara.use_state(State(""))

    solara.use_memo(lambda: controller.rand_select(state), [state])

    css = """
    .main {
        height: 100vh;
        padding-bottom: 50px;
        padding-top: 50px;
    }
    .image {
        max-height: 100%;
        border-radius: 10px;
    }
    .div-image > div {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    """

    solara.Style(css)

    with solara.VBox(classes=["main"]):
        with solara.Column(
                style="height: 100; align-items: center;width: 100%;row-gap: 50px;"):
            with solara.Row(style="width: 80%; justify-content: center;"):
                UserNameInput(state)

                # dataset
                with solara.Column(style=style_input + "row-gap: 0px;"):
                    solara.Select(label="Select a dataset",
                                  values=controller.get_datasets(),
                                  on_value=lambda dataset: controller.set_dataset(dataset, state),
                                  )
                    if state.get_error(StateErrorType.DATASET):
                        solara.Text(state.get_error(StateErrorType.DATASET),
                                    style=style_err_label)

            with solara.Column(
                    style="align-items: center; height: 70vh; justify-content: center;"):
                solara.Button("Skip", on_click=lambda: set_state(state.copy()))

                with solara.Row(style=style_row + "height: 90%; width: 90%;"):
                    ImageColumn("A", state, set_state)
                    ImageColumn("B", state, set_state)



@solara.component
def SelectButton(option: str, state: State, set_state: Callable):
    def on_click():
        if controller.select_image(option, state):
            set_state(State(state.user_name))
        else:
            set_state(state.copy())

    solara.Button(f"Select {option}", on_click=on_click)


@solara.component
def Image(image):
    with solara.Div(style="height: 85%; display: flex; justify-content: center; align-items: center;",
                    classes=["div-image"]):
        solara.Image(image, width="inherit", classes=["image"])


@solara.component
def SelectButton(option: str, state: State, set_state: Callable):
    def on_click():
        set_state(controller.select_image(option, state))

    solara.Button(f"Select {option}", on_click=on_click)


@solara.component
def ImageColumn(option: str, state: State, set_state: Callable):
    with solara.Column(style=style_col):
        SelectButton(option, state, set_state)
        Image(controller.get_image(option, state))


@solara.component
def UserNameInput(state: State):
    with solara.Column(style=style_input):
        solara.InputText("Enter your name:",
                         value=state.user_name,
                         on_value=state.set_user_name,
                         error=state.get_error(StateErrorType.USER_NAME),
                         )

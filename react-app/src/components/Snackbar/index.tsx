import {Snackbar} from "@mui/material";
import {Alert} from "@mui/lab";
import {Loader} from "react-feather";

interface SnackbarProps {
    msg: string;
    isLoading: boolean;
    show: boolean,
}

export function CustomSnackbar(props: {
    showSnackbar: SnackbarProps
    onClose: () => void
}) {
    return <Snackbar
        open={props.showSnackbar.show}
        autoHideDuration={props.showSnackbar.isLoading ? null : 5000}
        onClose={props.onClose}
    >
        {
            (props.showSnackbar.isLoading) ?
                (<Alert severity="info" icon={false}>
                    <div style={{width: "30px"}} className="flex justify-center items-center">
                        <Loader className="animate-spin" size={20}/>
                    </div>
                </Alert>) :
                (<Alert onClose={props.onClose} severity="success">
                    {props.showSnackbar.msg}
                </Alert>)
        }
    </Snackbar>;
}
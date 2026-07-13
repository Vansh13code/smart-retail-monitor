import DashboardLayout from "../layouts/DashboardLayout";
import AnalysisPage from "../components/AnalysisPage";

export default function OCR(){

return(

<DashboardLayout>

<AnalysisPage

title="OCR Module"

endpoint="/ocr/"

/>

</DashboardLayout>

)

}
